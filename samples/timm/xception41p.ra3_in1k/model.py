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
        L_self_modules_stem_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_0_modules_shortcut_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_1_modules_shortcut_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_2_modules_shortcut_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_11_modules_shortcut_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_stem_modules_1_parameters_weight_ = (
            L_self_modules_stem_modules_1_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_0_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_0_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_0_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_0_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_0_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_0_modules_shortcut_parameters_weight_ = (
            L_self_modules_blocks_modules_0_modules_shortcut_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_1_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_1_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_1_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_1_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_1_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_1_modules_shortcut_parameters_weight_ = (
            L_self_modules_blocks_modules_1_modules_shortcut_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_2_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_2_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_2_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_2_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_2_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_2_modules_shortcut_parameters_weight_ = (
            L_self_modules_blocks_modules_2_modules_shortcut_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_3_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_3_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_3_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_3_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_3_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_3_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_3_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_4_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_4_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_4_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_4_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_4_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_4_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_4_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_5_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_5_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_5_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_5_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_5_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_5_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_5_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_6_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_6_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_6_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_6_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_6_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_6_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_6_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_7_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_7_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_7_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_7_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_7_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_7_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_7_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_8_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_8_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_8_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_8_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_8_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_8_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_8_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_9_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_9_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_9_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_9_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_9_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_9_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_9_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_10_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_10_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_10_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_10_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_10_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_10_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_10_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_11_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_11_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_11_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_11_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_11_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_11_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_11_modules_shortcut_parameters_weight_ = (
            L_self_modules_blocks_modules_11_modules_shortcut_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_12_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_12_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_12_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_12_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_12_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_12_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
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
            1e-05,
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
        input_1 = torch.conv2d(
            x_2,
            l_self_modules_stem_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_stem_modules_1_parameters_weight_ = None
        x_3 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_blocks_modules_0_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_blocks_modules_0_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_0_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_0_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_0_modules_norm_parameters_bias_ = None
        x_4 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            64,
        )
        l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_blocks_modules_0_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        x_8 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_blocks_modules_0_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            128,
        )
        x_12 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_blocks_modules_0_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        conv2d_8 = torch.conv2d(
            x_4,
            l_self_modules_blocks_modules_0_modules_shortcut_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_blocks_modules_0_modules_shortcut_parameters_weight_ = None
        x_15 = x_14 + conv2d_8
        x_14 = conv2d_8 = None
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_blocks_modules_1_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_blocks_modules_1_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_1_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_1_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_1_modules_norm_parameters_bias_ = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            128,
        )
        l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_18 = l_self_modules_blocks_modules_1_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_22 = torch.conv2d(
            x_21,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        x_21 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_blocks_modules_1_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            256,
        )
        x_25 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_27 = torch.conv2d(
            x_26,
            l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_blocks_modules_1_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        conv2d_15 = torch.conv2d(
            x_17,
            l_self_modules_blocks_modules_1_modules_shortcut_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_17 = (
            l_self_modules_blocks_modules_1_modules_shortcut_parameters_weight_
        ) = None
        x_28 = x_27 + conv2d_15
        x_27 = conv2d_15 = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_blocks_modules_2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_blocks_modules_2_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_2_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_2_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_2_modules_norm_parameters_bias_ = None
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            256,
        )
        l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_blocks_modules_2_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_34 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_blocks_modules_2_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_38 = torch.nn.functional.relu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            728,
        )
        x_38 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_blocks_modules_2_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        conv2d_22 = torch.conv2d(
            x_30,
            l_self_modules_blocks_modules_2_modules_shortcut_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = (
            l_self_modules_blocks_modules_2_modules_shortcut_parameters_weight_
        ) = None
        x_41 = x_40 + conv2d_22
        x_40 = conv2d_22 = None
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_blocks_modules_3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = (
            l_self_modules_blocks_modules_3_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_3_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_3_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_3_modules_norm_parameters_bias_ = None
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = l_self_modules_blocks_modules_3_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_47,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_47 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_48 = l_self_modules_blocks_modules_3_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_51 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_blocks_modules_3_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_54 = x_53 + x_43
        x_53 = x_43 = None
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_blocks_modules_4_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_blocks_modules_4_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_4_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_4_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_4_modules_norm_parameters_bias_ = None
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_blocks_modules_4_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_60 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_blocks_modules_4_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_64 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_66 = torch.conv2d(
            x_65,
            l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_65 = l_self_modules_blocks_modules_4_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_67 = x_66 + x_56
        x_66 = x_56 = None
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_blocks_modules_5_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = (
            l_self_modules_blocks_modules_5_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_5_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_5_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_5_modules_norm_parameters_bias_ = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_blocks_modules_5_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_71 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_73 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = l_self_modules_blocks_modules_5_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_77 = torch.nn.functional.relu(x_76, inplace=True)
        x_76 = None
        x_78 = torch.conv2d(
            x_77,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_77 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_78 = l_self_modules_blocks_modules_5_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_80 = x_79 + x_69
        x_79 = x_69 = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_blocks_modules_6_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_blocks_modules_6_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_6_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_6_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_6_modules_norm_parameters_bias_ = None
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_blocks_modules_6_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_86 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_87 = l_self_modules_blocks_modules_6_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_90 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_blocks_modules_6_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_93 = x_92 + x_82
        x_92 = x_82 = None
        x_94 = torch.nn.functional.batch_norm(
            x_93,
            l_self_modules_blocks_modules_7_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_93 = (
            l_self_modules_blocks_modules_7_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_7_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_7_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_7_modules_norm_parameters_bias_ = None
        x_95 = torch.nn.functional.relu(x_94, inplace=True)
        x_94 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = l_self_modules_blocks_modules_7_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        x_100 = torch.conv2d(
            x_99,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_99 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_blocks_modules_7_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_103 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_blocks_modules_7_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_106 = x_105 + x_95
        x_105 = x_95 = None
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_blocks_modules_8_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = (
            l_self_modules_blocks_modules_8_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_8_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_8_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_8_modules_norm_parameters_bias_ = None
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_blocks_modules_8_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_112 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_blocks_modules_8_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_116 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_118 = torch.conv2d(
            x_117,
            l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_117 = l_self_modules_blocks_modules_8_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_119 = x_118 + x_108
        x_118 = x_108 = None
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_blocks_modules_9_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_blocks_modules_9_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_9_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_9_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_9_modules_norm_parameters_bias_ = None
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_123 = torch.conv2d(
            x_122,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_122 = l_self_modules_blocks_modules_9_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_123 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_125 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = l_self_modules_blocks_modules_9_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_129 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_blocks_modules_9_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_132 = x_131 + x_121
        x_131 = x_121 = None
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_blocks_modules_10_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_blocks_modules_10_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_10_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_10_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_10_modules_norm_parameters_bias_ = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_135 = l_self_modules_blocks_modules_10_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_138 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_blocks_modules_10_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_142 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_blocks_modules_10_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_145 = x_144 + x_134
        x_144 = x_134 = None
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_blocks_modules_11_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_blocks_modules_11_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_11_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_11_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_11_modules_norm_parameters_bias_ = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_148 = l_self_modules_blocks_modules_11_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_151 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_blocks_modules_11_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_154 = torch.nn.functional.batch_norm(
            x_153,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_153 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1024,
        )
        x_155 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        conv2d_77 = torch.conv2d(
            x_147,
            l_self_modules_blocks_modules_11_modules_shortcut_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = (
            l_self_modules_blocks_modules_11_modules_shortcut_parameters_weight_
        ) = None
        x_158 = x_157 + conv2d_77
        x_157 = conv2d_77 = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_158 = (
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_12_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_12_modules_norm_parameters_bias_ = None
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_160 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_164 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_165 = l_self_modules_blocks_modules_12_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_168 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_blocks_modules_12_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.nn.functional.adaptive_avg_pool2d(x_171, 1)
        x_171 = None
        x_173 = x_172.flatten(1, -1)
        x_172 = None
        x_174 = torch.nn.functional.dropout(x_173, 0.0, False, False)
        x_173 = None
        x_175 = torch._C._nn.linear(
            x_174,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_174 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_175,)
