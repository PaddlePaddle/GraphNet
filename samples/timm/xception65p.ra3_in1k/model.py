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
        L_self_modules_blocks_modules_13_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_19_modules_shortcut_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_: torch.nn.parameter.Parameter,
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
        l_self_modules_blocks_modules_13_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_13_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_13_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_13_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_13_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_13_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_13_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_13_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_14_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_14_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_14_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_14_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_14_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_14_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_14_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_15_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_15_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_15_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_15_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_15_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_15_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_15_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_16_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_16_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_16_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_16_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_16_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_16_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_16_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_17_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_17_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_17_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_17_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_17_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_17_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_17_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_18_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_18_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_18_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_18_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_18_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_18_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_18_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_19_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_19_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_19_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_19_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_19_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_19_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_19_modules_shortcut_parameters_weight_ = (
            L_self_modules_blocks_modules_19_modules_shortcut_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm_buffers_running_mean_ = (
            L_self_modules_blocks_modules_20_modules_norm_buffers_running_mean_
        )
        l_self_modules_blocks_modules_20_modules_norm_buffers_running_var_ = (
            L_self_modules_blocks_modules_20_modules_norm_buffers_running_var_
        )
        l_self_modules_blocks_modules_20_modules_norm_parameters_weight_ = (
            L_self_modules_blocks_modules_20_modules_norm_parameters_weight_
        )
        l_self_modules_blocks_modules_20_modules_norm_parameters_bias_ = (
            L_self_modules_blocks_modules_20_modules_norm_parameters_bias_
        )
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_mean_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_var_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_bias_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_bias_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_
        l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = L_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
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
            0.001,
        )
        x_153 = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_155 = torch.nn.functional.relu(x_154, inplace=True)
        x_154 = None
        x_156 = torch.conv2d(
            x_155,
            l_self_modules_blocks_modules_11_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
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
        x_158 = x_157 + x_147
        x_157 = x_147 = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_12_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_12_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_12_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
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
            728,
        )
        l_self_modules_blocks_modules_12_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
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
            0.001,
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
            728,
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
            0.001,
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
            728,
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
        x_171 = x_170 + x_160
        x_170 = x_160 = None
        x_172 = torch.nn.functional.batch_norm(
            x_171,
            l_self_modules_blocks_modules_13_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_171 = (
            l_self_modules_blocks_modules_13_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_13_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_13_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_13_modules_norm_parameters_bias_ = None
        x_173 = torch.nn.functional.relu(x_172, inplace=True)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_blocks_modules_13_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_175 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_177,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_177 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_blocks_modules_13_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_179 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_181 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_183 = torch.conv2d(
            x_182,
            l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_182 = l_self_modules_blocks_modules_13_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_184 = x_183 + x_173
        x_183 = x_173 = None
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_blocks_modules_14_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_184 = (
            l_self_modules_blocks_modules_14_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_14_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_14_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_14_modules_norm_parameters_bias_ = None
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_blocks_modules_14_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_188 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_190 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = l_self_modules_blocks_modules_14_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_192 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_194 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_195 = l_self_modules_blocks_modules_14_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_197 = x_196 + x_186
        x_196 = x_186 = None
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_blocks_modules_15_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_197 = (
            l_self_modules_blocks_modules_15_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_15_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_15_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_15_modules_norm_parameters_bias_ = None
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        x_200 = torch.conv2d(
            x_199,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_200 = l_self_modules_blocks_modules_15_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_201 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_203 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_205 = torch.conv2d(
            x_204,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_204 = l_self_modules_blocks_modules_15_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_205 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_207 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_blocks_modules_15_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_210 = x_209 + x_199
        x_209 = x_199 = None
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_blocks_modules_16_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_210 = (
            l_self_modules_blocks_modules_16_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_16_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_16_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_16_modules_norm_parameters_bias_ = None
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = l_self_modules_blocks_modules_16_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_214 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_216 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_217 = l_self_modules_blocks_modules_16_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_218 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_220 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_222 = torch.conv2d(
            x_221,
            l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_221 = l_self_modules_blocks_modules_16_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_223 = x_222 + x_212
        x_222 = x_212 = None
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_blocks_modules_17_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_223 = (
            l_self_modules_blocks_modules_17_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_17_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_17_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_17_modules_norm_parameters_bias_ = None
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_blocks_modules_17_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_227 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_229 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_blocks_modules_17_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_231 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_233 = torch.nn.functional.relu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_233 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_blocks_modules_17_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_236 = x_235 + x_225
        x_235 = x_225 = None
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_blocks_modules_18_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_236 = (
            l_self_modules_blocks_modules_18_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_18_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_18_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_18_modules_norm_parameters_bias_ = None
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_240 = torch.conv2d(
            x_239,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_239 = l_self_modules_blocks_modules_18_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_241 = torch.nn.functional.batch_norm(
            x_240,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_240 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_242 = torch.nn.functional.relu(x_241, inplace=True)
        x_241 = None
        x_243 = torch.conv2d(
            x_242,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_242 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_244 = torch.conv2d(
            x_243,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_243 = l_self_modules_blocks_modules_18_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_245 = torch.nn.functional.batch_norm(
            x_244,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_244 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_246 = torch.nn.functional.relu(x_245, inplace=True)
        x_245 = None
        x_247 = torch.conv2d(
            x_246,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_246 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_blocks_modules_18_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_249 = x_248 + x_238
        x_248 = x_238 = None
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_blocks_modules_19_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_249 = (
            l_self_modules_blocks_modules_19_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_19_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_19_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_19_modules_norm_parameters_bias_ = None
        x_251 = torch.nn.functional.relu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (
            None
        )
        x_253 = torch.conv2d(
            x_252,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_252 = l_self_modules_blocks_modules_19_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_254 = torch.nn.functional.batch_norm(
            x_253,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_253 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            728,
        )
        x_255 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_256 = l_self_modules_blocks_modules_19_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_257 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1024,
        )
        x_259 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_261 = torch.conv2d(
            x_260,
            l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_260 = l_self_modules_blocks_modules_19_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        conv2d_125 = torch.conv2d(
            x_251,
            l_self_modules_blocks_modules_19_modules_shortcut_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = (
            l_self_modules_blocks_modules_19_modules_shortcut_parameters_weight_
        ) = None
        x_262 = x_261 + conv2d_125
        x_261 = conv2d_125 = None
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_blocks_modules_20_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_262 = (
            l_self_modules_blocks_modules_20_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_blocks_modules_20_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_blocks_modules_20_modules_norm_parameters_weight_
        ) = l_self_modules_blocks_modules_20_modules_norm_parameters_bias_ = None
        x_264 = torch.nn.functional.relu(x_263, inplace=True)
        x_263 = None
        x_265 = torch.conv2d(
            x_264,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1024,
        )
        x_264 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_dw_parameters_weight_ = (None)
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_blocks_modules_20_modules_stack_modules_conv1_modules_conv_pw_parameters_weight_ = (None)
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_266 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_norm_parameters_bias_ = (None)
        x_268 = torch.nn.functional.relu(x_267, inplace=True)
        x_267 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_268 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_dw_parameters_weight_ = (None)
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_blocks_modules_20_modules_stack_modules_conv2_modules_conv_pw_parameters_weight_ = (None)
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_mean_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_var_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_weight_,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_270 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_mean_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_buffers_running_var_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_weight_ = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_norm_parameters_bias_ = (None)
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1536,
        )
        x_272 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_dw_parameters_weight_ = (None)
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_blocks_modules_20_modules_stack_modules_conv3_modules_conv_pw_parameters_weight_ = (None)
        x_275 = torch.nn.functional.relu(x_274, inplace=True)
        x_274 = None
        x_276 = torch.nn.functional.adaptive_avg_pool2d(x_275, 1)
        x_275 = None
        x_277 = x_276.flatten(1, -1)
        x_276 = None
        x_278 = torch.nn.functional.dropout(x_277, 0.0, False, False)
        x_277 = None
        x_279 = torch._C._nn.linear(
            x_278,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_278 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_279,)
