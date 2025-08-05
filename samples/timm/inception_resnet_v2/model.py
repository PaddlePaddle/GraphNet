import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv2d_1a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_conv2d_1a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_1a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_1a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_1a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_2a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_2a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_2b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_2b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_2b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_4a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_4a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_4a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_4a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_4a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch3_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_0_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_1_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_2_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_3_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_4_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_5_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_6_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_7_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_8_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_modules_9_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_0_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_1_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_2_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_3_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_4_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_5_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_6_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_7_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_8_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_9_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_10_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_11_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_12_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_13_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_14_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_15_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_16_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_17_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_18_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_1_modules_19_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_0_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_1_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_2_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_3_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_4_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_5_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_6_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_7_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_repeat_2_modules_8_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_branch0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_branch0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_conv2d_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_block8_modules_conv2d_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_7b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_7b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv2d_7b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_conv2d_7b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv2d_7b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_classif_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_classif_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv2d_1a_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_1a_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv2d_1a_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_1a_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_1a_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_1a_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_1a_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_1a_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_1a_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_1a_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_2a_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_2a_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_2a_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_2a_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_2a_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_2a_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_2a_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_2a_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_2a_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_2a_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_2b_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_2b_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_2b_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_2b_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_2b_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_2b_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_2b_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_2b_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_2b_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_2b_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_3b_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_3b_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_3b_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_3b_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_3b_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_3b_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_3b_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_3b_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_3b_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_3b_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_4a_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_4a_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_4a_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_4a_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_4a_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_4a_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_4a_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_4a_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_4a_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_4a_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch0_modules_conv_parameters_weight_ = (
            L_self_modules_mixed_5b_modules_branch0_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_mean_ = (
            L_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_var_ = (
            L_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_weight_ = (
            L_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_5b_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_5b_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_5b_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_mixed_5b_modules_branch3_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_5b_modules_branch3_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_0_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_0_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_0_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_0_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_1_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_1_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_1_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_1_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_1_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_2_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_2_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_2_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_2_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_2_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_3_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_3_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_3_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_3_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_3_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_4_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_4_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_4_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_4_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_4_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_5_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_5_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_5_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_5_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_5_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_6_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_6_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_6_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_6_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_6_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_7_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_7_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_7_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_7_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_7_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_8_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_8_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_8_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_8_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_8_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_modules_9_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_modules_9_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_modules_9_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_modules_9_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_modules_9_modules_conv2d_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch0_modules_conv_parameters_weight_ = (
            L_self_modules_mixed_6a_modules_branch0_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_mean_ = (
            L_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_var_ = (
            L_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_weight_ = (
            L_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_6a_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_6a_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_0_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_0_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_1_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_1_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_2_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_2_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_3_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_3_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_4_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_4_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_5_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_5_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_6_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_6_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_7_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_7_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_8_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_8_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_9_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_9_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_10_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_10_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_11_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_11_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_12_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_12_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_13_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_13_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_14_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_14_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_15_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_15_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_16_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_16_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_17_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_17_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_18_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_18_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_1_modules_19_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_1_modules_19_modules_conv2d_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch0_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch0_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch0_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_0_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch2_modules_1_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_bias_
        l_self_modules_mixed_7a_modules_branch2_modules_2_modules_conv_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_2_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_var_ = L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_weight_ = L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_weight_
        l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_bias_ = L_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_0_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_0_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_1_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_1_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_2_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_2_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_3_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_3_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_4_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_4_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_5_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_5_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_6_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_6_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_7_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_7_modules_conv2d_parameters_bias_
        )
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_ = L_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_
        l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_weight_ = (
            L_self_modules_repeat_2_modules_8_modules_conv2d_parameters_weight_
        )
        l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_bias_ = (
            L_self_modules_repeat_2_modules_8_modules_conv2d_parameters_bias_
        )
        l_self_modules_block8_modules_branch0_modules_conv_parameters_weight_ = (
            L_self_modules_block8_modules_branch0_modules_conv_parameters_weight_
        )
        l_self_modules_block8_modules_branch0_modules_bn_buffers_running_mean_ = (
            L_self_modules_block8_modules_branch0_modules_bn_buffers_running_mean_
        )
        l_self_modules_block8_modules_branch0_modules_bn_buffers_running_var_ = (
            L_self_modules_block8_modules_branch0_modules_bn_buffers_running_var_
        )
        l_self_modules_block8_modules_branch0_modules_bn_parameters_weight_ = (
            L_self_modules_block8_modules_branch0_modules_bn_parameters_weight_
        )
        l_self_modules_block8_modules_branch0_modules_bn_parameters_bias_ = (
            L_self_modules_block8_modules_branch0_modules_bn_parameters_bias_
        )
        l_self_modules_block8_modules_branch1_modules_0_modules_conv_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_0_modules_conv_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = L_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_var_
        l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_bias_ = (
            L_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_bias_
        )
        l_self_modules_block8_modules_branch1_modules_1_modules_conv_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_1_modules_conv_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = L_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_var_
        l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_bias_ = (
            L_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_bias_
        )
        l_self_modules_block8_modules_branch1_modules_2_modules_conv_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_2_modules_conv_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = L_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_mean_
        l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = L_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_var_
        l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_weight_ = L_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_weight_
        l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_bias_ = (
            L_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_bias_
        )
        l_self_modules_block8_modules_conv2d_parameters_weight_ = (
            L_self_modules_block8_modules_conv2d_parameters_weight_
        )
        l_self_modules_block8_modules_conv2d_parameters_bias_ = (
            L_self_modules_block8_modules_conv2d_parameters_bias_
        )
        l_self_modules_conv2d_7b_modules_conv_parameters_weight_ = (
            L_self_modules_conv2d_7b_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_7b_modules_bn_buffers_running_mean_ = (
            L_self_modules_conv2d_7b_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_7b_modules_bn_buffers_running_var_ = (
            L_self_modules_conv2d_7b_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_7b_modules_bn_parameters_weight_ = (
            L_self_modules_conv2d_7b_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_7b_modules_bn_parameters_bias_ = (
            L_self_modules_conv2d_7b_modules_bn_parameters_bias_
        )
        l_self_modules_classif_parameters_weight_ = (
            L_self_modules_classif_parameters_weight_
        )
        l_self_modules_classif_parameters_bias_ = (
            L_self_modules_classif_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv2d_1a_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv2d_1a_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_conv2d_1a_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_1a_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_1a_modules_bn_parameters_weight_,
            l_self_modules_conv2d_1a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x = (
            l_self_modules_conv2d_1a_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_1a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_1a_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_1a_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_conv2d_2a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_conv2d_2a_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_conv2d_2a_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_2a_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_2a_modules_bn_parameters_weight_,
            l_self_modules_conv2d_2a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_3 = (
            l_self_modules_conv2d_2a_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_2a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_2a_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_2a_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_conv2d_2b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_conv2d_2b_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_conv2d_2b_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_2b_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_2b_modules_bn_parameters_weight_,
            l_self_modules_conv2d_2b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_6 = (
            l_self_modules_conv2d_2b_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_2b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_2b_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_2b_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_conv2d_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_conv2d_3b_modules_conv_parameters_weight_ = None
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_conv2d_3b_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_3b_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_3b_modules_bn_parameters_weight_,
            l_self_modules_conv2d_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_10 = (
            l_self_modules_conv2d_3b_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_3b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_3b_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_3b_modules_bn_parameters_bias_ = None
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_conv2d_4a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_conv2d_4a_modules_conv_parameters_weight_ = None
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_conv2d_4a_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_4a_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_4a_modules_bn_parameters_weight_,
            l_self_modules_conv2d_4a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_13 = (
            l_self_modules_conv2d_4a_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_4a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_4a_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_4a_modules_bn_parameters_bias_ = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.nn.functional.max_pool2d(
            x_15, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch0_modules_conv_parameters_weight_ = None
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_17 = (
            l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5b_modules_branch0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_5b_modules_branch0_modules_bn_parameters_bias_ = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_20 = l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_mixed_5b_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_mixed_5b_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_23 = l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_26 = l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_mixed_5b_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_mixed_5b_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_29 = l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_mixed_5b_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_mixed_5b_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_32 = l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        input_1 = torch._C._nn.avg_pool2d(x_16, 3, 1, 1, False, False, None)
        x_16 = None
        x_35 = torch.conv2d(
            input_1,
            l_self_modules_mixed_5b_modules_branch3_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = l_self_modules_mixed_5b_modules_branch3_modules_1_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_35 = l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_5b_modules_branch3_modules_1_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        out = torch.cat((x_19, x_25, x_34, x_37), 1)
        x_19 = x_25 = x_34 = x_37 = None
        x_38 = torch.conv2d(
            out,
            l_self_modules_repeat_modules_0_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_0_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_38 = l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_0_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            out,
            l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_41 = l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_44 = l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            out,
            l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_47 = l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_0_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_50 = l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_0_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_53 = l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_0_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        out_1 = torch.cat((x_40, x_46, x_55), 1)
        x_40 = x_46 = x_55 = None
        out_2 = torch.conv2d(
            out_1,
            l_self_modules_repeat_modules_0_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_0_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_1 = (
            l_self_modules_repeat_modules_0_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_0_modules_conv2d_parameters_bias_ = None
        mul = out_2 * 0.17
        out_2 = None
        out_3 = mul + out
        mul = out = None
        out_4 = torch.nn.functional.relu(out_3, inplace=False)
        out_3 = None
        x_56 = torch.conv2d(
            out_4,
            l_self_modules_repeat_modules_1_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_1_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_56 = l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_1_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            out_4,
            l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_59 = l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_62 = l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            out_4,
            l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_65 = l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_1_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_68 = l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_1_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_72 = torch.nn.functional.batch_norm(
            x_71,
            l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_71 = l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_1_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        out_5 = torch.cat((x_58, x_64, x_73), 1)
        x_58 = x_64 = x_73 = None
        out_6 = torch.conv2d(
            out_5,
            l_self_modules_repeat_modules_1_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_1_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_5 = (
            l_self_modules_repeat_modules_1_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_1_modules_conv2d_parameters_bias_ = None
        mul_1 = out_6 * 0.17
        out_6 = None
        out_7 = mul_1 + out_4
        mul_1 = out_4 = None
        out_8 = torch.nn.functional.relu(out_7, inplace=False)
        out_7 = None
        x_74 = torch.conv2d(
            out_8,
            l_self_modules_repeat_modules_2_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_2_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_74 = l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_2_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            out_8,
            l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_77 = l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_80 = l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            out_8,
            l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_83 = l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_2_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_86 = l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_2_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_89 = l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_2_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        out_9 = torch.cat((x_76, x_82, x_91), 1)
        x_76 = x_82 = x_91 = None
        out_10 = torch.conv2d(
            out_9,
            l_self_modules_repeat_modules_2_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_2_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_9 = (
            l_self_modules_repeat_modules_2_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_2_modules_conv2d_parameters_bias_ = None
        mul_2 = out_10 * 0.17
        out_10 = None
        out_11 = mul_2 + out_8
        mul_2 = out_8 = None
        out_12 = torch.nn.functional.relu(out_11, inplace=False)
        out_11 = None
        x_92 = torch.conv2d(
            out_12,
            l_self_modules_repeat_modules_3_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_3_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_92 = l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_3_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            out_12,
            l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_95 = l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_98 = l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            out_12,
            l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_101 = l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_3_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_104 = l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_3_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_107 = l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_3_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        out_13 = torch.cat((x_94, x_100, x_109), 1)
        x_94 = x_100 = x_109 = None
        out_14 = torch.conv2d(
            out_13,
            l_self_modules_repeat_modules_3_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_3_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_13 = (
            l_self_modules_repeat_modules_3_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_3_modules_conv2d_parameters_bias_ = None
        mul_3 = out_14 * 0.17
        out_14 = None
        out_15 = mul_3 + out_12
        mul_3 = out_12 = None
        out_16 = torch.nn.functional.relu(out_15, inplace=False)
        out_15 = None
        x_110 = torch.conv2d(
            out_16,
            l_self_modules_repeat_modules_4_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_4_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_110 = l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_4_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            out_16,
            l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_113 = l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_117 = torch.nn.functional.batch_norm(
            x_116,
            l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_116 = l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        x_119 = torch.conv2d(
            out_16,
            l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_119 = l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_4_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_122 = l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_4_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_125 = l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_4_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        out_17 = torch.cat((x_112, x_118, x_127), 1)
        x_112 = x_118 = x_127 = None
        out_18 = torch.conv2d(
            out_17,
            l_self_modules_repeat_modules_4_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_4_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_17 = (
            l_self_modules_repeat_modules_4_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_4_modules_conv2d_parameters_bias_ = None
        mul_4 = out_18 * 0.17
        out_18 = None
        out_19 = mul_4 + out_16
        mul_4 = out_16 = None
        out_20 = torch.nn.functional.relu(out_19, inplace=False)
        out_19 = None
        x_128 = torch.conv2d(
            out_20,
            l_self_modules_repeat_modules_5_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_5_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_128 = l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_5_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            out_20,
            l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_131 = l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_134 = l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            out_20,
            l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_137 = l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_5_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_140 = l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_5_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_143 = l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_5_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        out_21 = torch.cat((x_130, x_136, x_145), 1)
        x_130 = x_136 = x_145 = None
        out_22 = torch.conv2d(
            out_21,
            l_self_modules_repeat_modules_5_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_5_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_21 = (
            l_self_modules_repeat_modules_5_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_5_modules_conv2d_parameters_bias_ = None
        mul_5 = out_22 * 0.17
        out_22 = None
        out_23 = mul_5 + out_20
        mul_5 = out_20 = None
        out_24 = torch.nn.functional.relu(out_23, inplace=False)
        out_23 = None
        x_146 = torch.conv2d(
            out_24,
            l_self_modules_repeat_modules_6_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_6_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_146 = l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_6_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_148 = torch.nn.functional.relu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            out_24,
            l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_149 = l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_151 = l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_152 = l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            out_24,
            l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_155 = l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_6_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_158 = l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_6_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_161 = l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_6_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        out_25 = torch.cat((x_148, x_154, x_163), 1)
        x_148 = x_154 = x_163 = None
        out_26 = torch.conv2d(
            out_25,
            l_self_modules_repeat_modules_6_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_6_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_25 = (
            l_self_modules_repeat_modules_6_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_6_modules_conv2d_parameters_bias_ = None
        mul_6 = out_26 * 0.17
        out_26 = None
        out_27 = mul_6 + out_24
        mul_6 = out_24 = None
        out_28 = torch.nn.functional.relu(out_27, inplace=False)
        out_27 = None
        x_164 = torch.conv2d(
            out_28,
            l_self_modules_repeat_modules_7_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_7_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_164 = l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_7_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            out_28,
            l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_167 = l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_169 = l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_170 = l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            out_28,
            l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_173 = l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_7_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_176 = l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_7_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_179 = l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_7_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        out_29 = torch.cat((x_166, x_172, x_181), 1)
        x_166 = x_172 = x_181 = None
        out_30 = torch.conv2d(
            out_29,
            l_self_modules_repeat_modules_7_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_7_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_29 = (
            l_self_modules_repeat_modules_7_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_7_modules_conv2d_parameters_bias_ = None
        mul_7 = out_30 * 0.17
        out_30 = None
        out_31 = mul_7 + out_28
        mul_7 = out_28 = None
        out_32 = torch.nn.functional.relu(out_31, inplace=False)
        out_31 = None
        x_182 = torch.conv2d(
            out_32,
            l_self_modules_repeat_modules_8_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_8_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_182 = l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_8_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.conv2d(
            out_32,
            l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_185 = l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_187 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_187 = l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_188 = l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_191 = torch.conv2d(
            out_32,
            l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_191 = l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_8_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_194 = l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_8_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_196 = torch.nn.functional.relu(x_195, inplace=True)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_198 = torch.nn.functional.batch_norm(
            x_197,
            l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_197 = l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_8_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_199 = torch.nn.functional.relu(x_198, inplace=True)
        x_198 = None
        out_33 = torch.cat((x_184, x_190, x_199), 1)
        x_184 = x_190 = x_199 = None
        out_34 = torch.conv2d(
            out_33,
            l_self_modules_repeat_modules_8_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_8_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_33 = (
            l_self_modules_repeat_modules_8_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_8_modules_conv2d_parameters_bias_ = None
        mul_8 = out_34 * 0.17
        out_34 = None
        out_35 = mul_8 + out_32
        mul_8 = out_32 = None
        out_36 = torch.nn.functional.relu(out_35, inplace=False)
        out_35 = None
        x_200 = torch.conv2d(
            out_36,
            l_self_modules_repeat_modules_9_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_9_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_201 = torch.nn.functional.batch_norm(
            x_200,
            l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_200 = l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_weight_ = (
            l_self_modules_repeat_modules_9_modules_branch0_modules_bn_parameters_bias_
        ) = None
        x_202 = torch.nn.functional.relu(x_201, inplace=True)
        x_201 = None
        x_203 = torch.conv2d(
            out_36,
            l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_203 = l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_207 = torch.nn.functional.batch_norm(
            x_206,
            l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_206 = l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_208 = torch.nn.functional.relu(x_207, inplace=True)
        x_207 = None
        x_209 = torch.conv2d(
            out_36,
            l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_209 = l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_9_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_212 = l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_9_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_214 = torch.nn.functional.relu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_214 = l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_216 = torch.nn.functional.batch_norm(
            x_215,
            l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_215 = l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_modules_9_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_217 = torch.nn.functional.relu(x_216, inplace=True)
        x_216 = None
        out_37 = torch.cat((x_202, x_208, x_217), 1)
        x_202 = x_208 = x_217 = None
        out_38 = torch.conv2d(
            out_37,
            l_self_modules_repeat_modules_9_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_modules_9_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_37 = (
            l_self_modules_repeat_modules_9_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_modules_9_modules_conv2d_parameters_bias_ = None
        mul_9 = out_38 * 0.17
        out_38 = None
        out_39 = mul_9 + out_36
        mul_9 = out_36 = None
        out_40 = torch.nn.functional.relu(out_39, inplace=False)
        out_39 = None
        x_218 = torch.conv2d(
            out_40,
            l_self_modules_mixed_6a_modules_branch0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6a_modules_branch0_modules_conv_parameters_weight_ = None
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_218 = (
            l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6a_modules_branch0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6a_modules_branch0_modules_bn_parameters_bias_ = None
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            out_40,
            l_self_modules_mixed_6a_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6a_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_221 = l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_6a_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_mixed_6a_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_mixed_6a_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_224 = l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_6a_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_mixed_6a_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_mixed_6a_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_227 = l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_mixed_6a_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x2 = torch.nn.functional.max_pool2d(
            out_40, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_40 = None
        out_41 = torch.cat((x_220, x_229, x2), 1)
        x_220 = x_229 = x2 = None
        x_230 = torch.conv2d(
            out_41,
            l_self_modules_repeat_1_modules_0_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_0_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_231 = torch.nn.functional.batch_norm(
            x_230,
            l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_230 = l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_0_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_232 = torch.nn.functional.relu(x_231, inplace=True)
        x_231 = None
        x_233 = torch.conv2d(
            out_41,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_234 = torch.nn.functional.batch_norm(
            x_233,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_233 = l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_235 = torch.nn.functional.relu(x_234, inplace=True)
        x_234 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_235 = l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_236 = l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_238 = torch.nn.functional.relu(x_237, inplace=True)
        x_237 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_238 = l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_239 = l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_241 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        out_42 = torch.cat((x_232, x_241), 1)
        x_232 = x_241 = None
        out_43 = torch.conv2d(
            out_42,
            l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_42 = (
            l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_0_modules_conv2d_parameters_bias_ = None
        mul_10 = out_43 * 0.1
        out_43 = None
        out_44 = mul_10 + out_41
        mul_10 = out_41 = None
        out_45 = torch.nn.functional.relu(out_44, inplace=False)
        out_44 = None
        x_242 = torch.conv2d(
            out_45,
            l_self_modules_repeat_1_modules_1_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_1_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_242 = l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_1_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            out_45,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_245 = l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_248 = l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_250 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        x_251 = torch.conv2d(
            x_250,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_250 = l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_251 = l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        out_46 = torch.cat((x_244, x_253), 1)
        x_244 = x_253 = None
        out_47 = torch.conv2d(
            out_46,
            l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_46 = (
            l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_1_modules_conv2d_parameters_bias_ = None
        mul_11 = out_47 * 0.1
        out_47 = None
        out_48 = mul_11 + out_45
        mul_11 = out_45 = None
        out_49 = torch.nn.functional.relu(out_48, inplace=False)
        out_48 = None
        x_254 = torch.conv2d(
            out_49,
            l_self_modules_repeat_1_modules_2_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_2_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_254 = l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_2_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            out_49,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_257 = l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_259,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_259 = l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_260 = l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_263 = l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        out_50 = torch.cat((x_256, x_265), 1)
        x_256 = x_265 = None
        out_51 = torch.conv2d(
            out_50,
            l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_50 = (
            l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_2_modules_conv2d_parameters_bias_ = None
        mul_12 = out_51 * 0.1
        out_51 = None
        out_52 = mul_12 + out_49
        mul_12 = out_49 = None
        out_53 = torch.nn.functional.relu(out_52, inplace=False)
        out_52 = None
        x_266 = torch.conv2d(
            out_53,
            l_self_modules_repeat_1_modules_3_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_3_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_266 = l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_3_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_268 = torch.nn.functional.relu(x_267, inplace=True)
        x_267 = None
        x_269 = torch.conv2d(
            out_53,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_270 = torch.nn.functional.batch_norm(
            x_269,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_269 = l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_271 = torch.nn.functional.relu(x_270, inplace=True)
        x_270 = None
        x_272 = torch.conv2d(
            x_271,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_271 = l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_273 = torch.nn.functional.batch_norm(
            x_272,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_272 = l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_274 = torch.nn.functional.relu(x_273, inplace=True)
        x_273 = None
        x_275 = torch.conv2d(
            x_274,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_274 = l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_276 = torch.nn.functional.batch_norm(
            x_275,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_275 = l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_277 = torch.nn.functional.relu(x_276, inplace=True)
        x_276 = None
        out_54 = torch.cat((x_268, x_277), 1)
        x_268 = x_277 = None
        out_55 = torch.conv2d(
            out_54,
            l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_54 = (
            l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_3_modules_conv2d_parameters_bias_ = None
        mul_13 = out_55 * 0.1
        out_55 = None
        out_56 = mul_13 + out_53
        mul_13 = out_53 = None
        out_57 = torch.nn.functional.relu(out_56, inplace=False)
        out_56 = None
        x_278 = torch.conv2d(
            out_57,
            l_self_modules_repeat_1_modules_4_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_4_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_279 = torch.nn.functional.batch_norm(
            x_278,
            l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_278 = l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_4_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_280 = torch.nn.functional.relu(x_279, inplace=True)
        x_279 = None
        x_281 = torch.conv2d(
            out_57,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_281 = l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_285 = torch.nn.functional.batch_norm(
            x_284,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_284 = l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_286 = torch.nn.functional.relu(x_285, inplace=True)
        x_285 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_286 = l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_287 = l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        out_58 = torch.cat((x_280, x_289), 1)
        x_280 = x_289 = None
        out_59 = torch.conv2d(
            out_58,
            l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_58 = (
            l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_4_modules_conv2d_parameters_bias_ = None
        mul_14 = out_59 * 0.1
        out_59 = None
        out_60 = mul_14 + out_57
        mul_14 = out_57 = None
        out_61 = torch.nn.functional.relu(out_60, inplace=False)
        out_60 = None
        x_290 = torch.conv2d(
            out_61,
            l_self_modules_repeat_1_modules_5_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_5_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_291 = torch.nn.functional.batch_norm(
            x_290,
            l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_290 = l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_5_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_292 = torch.nn.functional.relu(x_291, inplace=True)
        x_291 = None
        x_293 = torch.conv2d(
            out_61,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_294 = torch.nn.functional.batch_norm(
            x_293,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_293 = l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_295 = torch.nn.functional.relu(x_294, inplace=True)
        x_294 = None
        x_296 = torch.conv2d(
            x_295,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_295 = l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_297 = torch.nn.functional.batch_norm(
            x_296,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_296 = l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_298 = torch.nn.functional.relu(x_297, inplace=True)
        x_297 = None
        x_299 = torch.conv2d(
            x_298,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_298 = l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_300 = torch.nn.functional.batch_norm(
            x_299,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_299 = l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_301 = torch.nn.functional.relu(x_300, inplace=True)
        x_300 = None
        out_62 = torch.cat((x_292, x_301), 1)
        x_292 = x_301 = None
        out_63 = torch.conv2d(
            out_62,
            l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_62 = (
            l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_5_modules_conv2d_parameters_bias_ = None
        mul_15 = out_63 * 0.1
        out_63 = None
        out_64 = mul_15 + out_61
        mul_15 = out_61 = None
        out_65 = torch.nn.functional.relu(out_64, inplace=False)
        out_64 = None
        x_302 = torch.conv2d(
            out_65,
            l_self_modules_repeat_1_modules_6_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_6_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_303 = torch.nn.functional.batch_norm(
            x_302,
            l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_302 = l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_6_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_304 = torch.nn.functional.relu(x_303, inplace=True)
        x_303 = None
        x_305 = torch.conv2d(
            out_65,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_306 = torch.nn.functional.batch_norm(
            x_305,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_305 = l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_307 = torch.nn.functional.relu(x_306, inplace=True)
        x_306 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_308 = l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_312 = torch.nn.functional.batch_norm(
            x_311,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_311 = l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_313 = torch.nn.functional.relu(x_312, inplace=True)
        x_312 = None
        out_66 = torch.cat((x_304, x_313), 1)
        x_304 = x_313 = None
        out_67 = torch.conv2d(
            out_66,
            l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_66 = (
            l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_6_modules_conv2d_parameters_bias_ = None
        mul_16 = out_67 * 0.1
        out_67 = None
        out_68 = mul_16 + out_65
        mul_16 = out_65 = None
        out_69 = torch.nn.functional.relu(out_68, inplace=False)
        out_68 = None
        x_314 = torch.conv2d(
            out_69,
            l_self_modules_repeat_1_modules_7_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_7_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_315 = torch.nn.functional.batch_norm(
            x_314,
            l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_314 = l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_7_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_316 = torch.nn.functional.relu(x_315, inplace=True)
        x_315 = None
        x_317 = torch.conv2d(
            out_69,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_318 = torch.nn.functional.batch_norm(
            x_317,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_317 = l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_319 = torch.nn.functional.relu(x_318, inplace=True)
        x_318 = None
        x_320 = torch.conv2d(
            x_319,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_319 = l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_321 = torch.nn.functional.batch_norm(
            x_320,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_320 = l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_322 = torch.nn.functional.relu(x_321, inplace=True)
        x_321 = None
        x_323 = torch.conv2d(
            x_322,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_322 = l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_324 = torch.nn.functional.batch_norm(
            x_323,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_323 = l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_325 = torch.nn.functional.relu(x_324, inplace=True)
        x_324 = None
        out_70 = torch.cat((x_316, x_325), 1)
        x_316 = x_325 = None
        out_71 = torch.conv2d(
            out_70,
            l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_70 = (
            l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_7_modules_conv2d_parameters_bias_ = None
        mul_17 = out_71 * 0.1
        out_71 = None
        out_72 = mul_17 + out_69
        mul_17 = out_69 = None
        out_73 = torch.nn.functional.relu(out_72, inplace=False)
        out_72 = None
        x_326 = torch.conv2d(
            out_73,
            l_self_modules_repeat_1_modules_8_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_8_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_327 = torch.nn.functional.batch_norm(
            x_326,
            l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_326 = l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_8_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_328 = torch.nn.functional.relu(x_327, inplace=True)
        x_327 = None
        x_329 = torch.conv2d(
            out_73,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_330 = torch.nn.functional.batch_norm(
            x_329,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_329 = l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_331 = torch.nn.functional.relu(x_330, inplace=True)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_333 = torch.nn.functional.batch_norm(
            x_332,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_332 = l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_334 = torch.nn.functional.relu(x_333, inplace=True)
        x_333 = None
        x_335 = torch.conv2d(
            x_334,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_334 = l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_336 = torch.nn.functional.batch_norm(
            x_335,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_335 = l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_337 = torch.nn.functional.relu(x_336, inplace=True)
        x_336 = None
        out_74 = torch.cat((x_328, x_337), 1)
        x_328 = x_337 = None
        out_75 = torch.conv2d(
            out_74,
            l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_74 = (
            l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_8_modules_conv2d_parameters_bias_ = None
        mul_18 = out_75 * 0.1
        out_75 = None
        out_76 = mul_18 + out_73
        mul_18 = out_73 = None
        out_77 = torch.nn.functional.relu(out_76, inplace=False)
        out_76 = None
        x_338 = torch.conv2d(
            out_77,
            l_self_modules_repeat_1_modules_9_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_9_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_339 = torch.nn.functional.batch_norm(
            x_338,
            l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_338 = l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_9_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_340 = torch.nn.functional.relu(x_339, inplace=True)
        x_339 = None
        x_341 = torch.conv2d(
            out_77,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_342 = torch.nn.functional.batch_norm(
            x_341,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_341 = l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_343 = torch.nn.functional.relu(x_342, inplace=True)
        x_342 = None
        x_344 = torch.conv2d(
            x_343,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_343 = l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_345 = torch.nn.functional.batch_norm(
            x_344,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_344 = l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_346 = torch.nn.functional.relu(x_345, inplace=True)
        x_345 = None
        x_347 = torch.conv2d(
            x_346,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_346 = l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_348 = torch.nn.functional.batch_norm(
            x_347,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_347 = l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_9_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_349 = torch.nn.functional.relu(x_348, inplace=True)
        x_348 = None
        out_78 = torch.cat((x_340, x_349), 1)
        x_340 = x_349 = None
        out_79 = torch.conv2d(
            out_78,
            l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_78 = (
            l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_9_modules_conv2d_parameters_bias_ = None
        mul_19 = out_79 * 0.1
        out_79 = None
        out_80 = mul_19 + out_77
        mul_19 = out_77 = None
        out_81 = torch.nn.functional.relu(out_80, inplace=False)
        out_80 = None
        x_350 = torch.conv2d(
            out_81,
            l_self_modules_repeat_1_modules_10_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_10_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_351 = torch.nn.functional.batch_norm(
            x_350,
            l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_350 = l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_10_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_352 = torch.nn.functional.relu(x_351, inplace=True)
        x_351 = None
        x_353 = torch.conv2d(
            out_81,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_354 = torch.nn.functional.batch_norm(
            x_353,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_353 = l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_355 = torch.nn.functional.relu(x_354, inplace=True)
        x_354 = None
        x_356 = torch.conv2d(
            x_355,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_355 = l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_357 = torch.nn.functional.batch_norm(
            x_356,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_356 = l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_358 = torch.nn.functional.relu(x_357, inplace=True)
        x_357 = None
        x_359 = torch.conv2d(
            x_358,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_358 = l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_360 = torch.nn.functional.batch_norm(
            x_359,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_359 = l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_10_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_361 = torch.nn.functional.relu(x_360, inplace=True)
        x_360 = None
        out_82 = torch.cat((x_352, x_361), 1)
        x_352 = x_361 = None
        out_83 = torch.conv2d(
            out_82,
            l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_82 = (
            l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_10_modules_conv2d_parameters_bias_ = None
        mul_20 = out_83 * 0.1
        out_83 = None
        out_84 = mul_20 + out_81
        mul_20 = out_81 = None
        out_85 = torch.nn.functional.relu(out_84, inplace=False)
        out_84 = None
        x_362 = torch.conv2d(
            out_85,
            l_self_modules_repeat_1_modules_11_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_11_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_363 = torch.nn.functional.batch_norm(
            x_362,
            l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_362 = l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_11_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_364 = torch.nn.functional.relu(x_363, inplace=True)
        x_363 = None
        x_365 = torch.conv2d(
            out_85,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_366 = torch.nn.functional.batch_norm(
            x_365,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_365 = l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_367 = torch.nn.functional.relu(x_366, inplace=True)
        x_366 = None
        x_368 = torch.conv2d(
            x_367,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_367 = l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_369 = torch.nn.functional.batch_norm(
            x_368,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_368 = l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_370 = torch.nn.functional.relu(x_369, inplace=True)
        x_369 = None
        x_371 = torch.conv2d(
            x_370,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_370 = l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_372 = torch.nn.functional.batch_norm(
            x_371,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_371 = l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_11_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_373 = torch.nn.functional.relu(x_372, inplace=True)
        x_372 = None
        out_86 = torch.cat((x_364, x_373), 1)
        x_364 = x_373 = None
        out_87 = torch.conv2d(
            out_86,
            l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_86 = (
            l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_11_modules_conv2d_parameters_bias_ = None
        mul_21 = out_87 * 0.1
        out_87 = None
        out_88 = mul_21 + out_85
        mul_21 = out_85 = None
        out_89 = torch.nn.functional.relu(out_88, inplace=False)
        out_88 = None
        x_374 = torch.conv2d(
            out_89,
            l_self_modules_repeat_1_modules_12_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_12_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_375 = torch.nn.functional.batch_norm(
            x_374,
            l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_374 = l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_12_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_376 = torch.nn.functional.relu(x_375, inplace=True)
        x_375 = None
        x_377 = torch.conv2d(
            out_89,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_378 = torch.nn.functional.batch_norm(
            x_377,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_377 = l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_379 = torch.nn.functional.relu(x_378, inplace=True)
        x_378 = None
        x_380 = torch.conv2d(
            x_379,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_379 = l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_381 = torch.nn.functional.batch_norm(
            x_380,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_380 = l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_382 = torch.nn.functional.relu(x_381, inplace=True)
        x_381 = None
        x_383 = torch.conv2d(
            x_382,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_382 = l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_384 = torch.nn.functional.batch_norm(
            x_383,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_383 = l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_12_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_385 = torch.nn.functional.relu(x_384, inplace=True)
        x_384 = None
        out_90 = torch.cat((x_376, x_385), 1)
        x_376 = x_385 = None
        out_91 = torch.conv2d(
            out_90,
            l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_90 = (
            l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_12_modules_conv2d_parameters_bias_ = None
        mul_22 = out_91 * 0.1
        out_91 = None
        out_92 = mul_22 + out_89
        mul_22 = out_89 = None
        out_93 = torch.nn.functional.relu(out_92, inplace=False)
        out_92 = None
        x_386 = torch.conv2d(
            out_93,
            l_self_modules_repeat_1_modules_13_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_13_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_387 = torch.nn.functional.batch_norm(
            x_386,
            l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_386 = l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_13_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_388 = torch.nn.functional.relu(x_387, inplace=True)
        x_387 = None
        x_389 = torch.conv2d(
            out_93,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_390 = torch.nn.functional.batch_norm(
            x_389,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_389 = l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_391 = torch.nn.functional.relu(x_390, inplace=True)
        x_390 = None
        x_392 = torch.conv2d(
            x_391,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_391 = l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_393 = torch.nn.functional.batch_norm(
            x_392,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_392 = l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_394 = torch.nn.functional.relu(x_393, inplace=True)
        x_393 = None
        x_395 = torch.conv2d(
            x_394,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_394 = l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_396 = torch.nn.functional.batch_norm(
            x_395,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_395 = l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_13_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_397 = torch.nn.functional.relu(x_396, inplace=True)
        x_396 = None
        out_94 = torch.cat((x_388, x_397), 1)
        x_388 = x_397 = None
        out_95 = torch.conv2d(
            out_94,
            l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_94 = (
            l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_13_modules_conv2d_parameters_bias_ = None
        mul_23 = out_95 * 0.1
        out_95 = None
        out_96 = mul_23 + out_93
        mul_23 = out_93 = None
        out_97 = torch.nn.functional.relu(out_96, inplace=False)
        out_96 = None
        x_398 = torch.conv2d(
            out_97,
            l_self_modules_repeat_1_modules_14_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_14_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_399 = torch.nn.functional.batch_norm(
            x_398,
            l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_398 = l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_14_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_400 = torch.nn.functional.relu(x_399, inplace=True)
        x_399 = None
        x_401 = torch.conv2d(
            out_97,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_402 = torch.nn.functional.batch_norm(
            x_401,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_401 = l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_403 = torch.nn.functional.relu(x_402, inplace=True)
        x_402 = None
        x_404 = torch.conv2d(
            x_403,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_403 = l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_405 = torch.nn.functional.batch_norm(
            x_404,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_404 = l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_406 = torch.nn.functional.relu(x_405, inplace=True)
        x_405 = None
        x_407 = torch.conv2d(
            x_406,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_406 = l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_408 = torch.nn.functional.batch_norm(
            x_407,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_407 = l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_14_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_409 = torch.nn.functional.relu(x_408, inplace=True)
        x_408 = None
        out_98 = torch.cat((x_400, x_409), 1)
        x_400 = x_409 = None
        out_99 = torch.conv2d(
            out_98,
            l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_98 = (
            l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_14_modules_conv2d_parameters_bias_ = None
        mul_24 = out_99 * 0.1
        out_99 = None
        out_100 = mul_24 + out_97
        mul_24 = out_97 = None
        out_101 = torch.nn.functional.relu(out_100, inplace=False)
        out_100 = None
        x_410 = torch.conv2d(
            out_101,
            l_self_modules_repeat_1_modules_15_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_15_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_411 = torch.nn.functional.batch_norm(
            x_410,
            l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_410 = l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_15_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_412 = torch.nn.functional.relu(x_411, inplace=True)
        x_411 = None
        x_413 = torch.conv2d(
            out_101,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_414 = torch.nn.functional.batch_norm(
            x_413,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_413 = l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_415 = torch.nn.functional.relu(x_414, inplace=True)
        x_414 = None
        x_416 = torch.conv2d(
            x_415,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_415 = l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_417 = torch.nn.functional.batch_norm(
            x_416,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_416 = l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_418 = torch.nn.functional.relu(x_417, inplace=True)
        x_417 = None
        x_419 = torch.conv2d(
            x_418,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_418 = l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_420 = torch.nn.functional.batch_norm(
            x_419,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_419 = l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_15_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_421 = torch.nn.functional.relu(x_420, inplace=True)
        x_420 = None
        out_102 = torch.cat((x_412, x_421), 1)
        x_412 = x_421 = None
        out_103 = torch.conv2d(
            out_102,
            l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_102 = (
            l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_15_modules_conv2d_parameters_bias_ = None
        mul_25 = out_103 * 0.1
        out_103 = None
        out_104 = mul_25 + out_101
        mul_25 = out_101 = None
        out_105 = torch.nn.functional.relu(out_104, inplace=False)
        out_104 = None
        x_422 = torch.conv2d(
            out_105,
            l_self_modules_repeat_1_modules_16_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_16_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_423 = torch.nn.functional.batch_norm(
            x_422,
            l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_422 = l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_16_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_424 = torch.nn.functional.relu(x_423, inplace=True)
        x_423 = None
        x_425 = torch.conv2d(
            out_105,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_426 = torch.nn.functional.batch_norm(
            x_425,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_425 = l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_427 = torch.nn.functional.relu(x_426, inplace=True)
        x_426 = None
        x_428 = torch.conv2d(
            x_427,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_427 = l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_429 = torch.nn.functional.batch_norm(
            x_428,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_428 = l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_430 = torch.nn.functional.relu(x_429, inplace=True)
        x_429 = None
        x_431 = torch.conv2d(
            x_430,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_430 = l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_432 = torch.nn.functional.batch_norm(
            x_431,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_431 = l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_16_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_433 = torch.nn.functional.relu(x_432, inplace=True)
        x_432 = None
        out_106 = torch.cat((x_424, x_433), 1)
        x_424 = x_433 = None
        out_107 = torch.conv2d(
            out_106,
            l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_106 = (
            l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_16_modules_conv2d_parameters_bias_ = None
        mul_26 = out_107 * 0.1
        out_107 = None
        out_108 = mul_26 + out_105
        mul_26 = out_105 = None
        out_109 = torch.nn.functional.relu(out_108, inplace=False)
        out_108 = None
        x_434 = torch.conv2d(
            out_109,
            l_self_modules_repeat_1_modules_17_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_17_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_435 = torch.nn.functional.batch_norm(
            x_434,
            l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_434 = l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_17_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_436 = torch.nn.functional.relu(x_435, inplace=True)
        x_435 = None
        x_437 = torch.conv2d(
            out_109,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_438 = torch.nn.functional.batch_norm(
            x_437,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_437 = l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_439 = torch.nn.functional.relu(x_438, inplace=True)
        x_438 = None
        x_440 = torch.conv2d(
            x_439,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_439 = l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_441 = torch.nn.functional.batch_norm(
            x_440,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_440 = l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_442 = torch.nn.functional.relu(x_441, inplace=True)
        x_441 = None
        x_443 = torch.conv2d(
            x_442,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_442 = l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_444 = torch.nn.functional.batch_norm(
            x_443,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_443 = l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_17_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_445 = torch.nn.functional.relu(x_444, inplace=True)
        x_444 = None
        out_110 = torch.cat((x_436, x_445), 1)
        x_436 = x_445 = None
        out_111 = torch.conv2d(
            out_110,
            l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_110 = (
            l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_17_modules_conv2d_parameters_bias_ = None
        mul_27 = out_111 * 0.1
        out_111 = None
        out_112 = mul_27 + out_109
        mul_27 = out_109 = None
        out_113 = torch.nn.functional.relu(out_112, inplace=False)
        out_112 = None
        x_446 = torch.conv2d(
            out_113,
            l_self_modules_repeat_1_modules_18_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_18_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_447 = torch.nn.functional.batch_norm(
            x_446,
            l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_446 = l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_18_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_448 = torch.nn.functional.relu(x_447, inplace=True)
        x_447 = None
        x_449 = torch.conv2d(
            out_113,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_450 = torch.nn.functional.batch_norm(
            x_449,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_449 = l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_451 = torch.nn.functional.relu(x_450, inplace=True)
        x_450 = None
        x_452 = torch.conv2d(
            x_451,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_451 = l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_453 = torch.nn.functional.batch_norm(
            x_452,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_452 = l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_454 = torch.nn.functional.relu(x_453, inplace=True)
        x_453 = None
        x_455 = torch.conv2d(
            x_454,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_454 = l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_456 = torch.nn.functional.batch_norm(
            x_455,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_455 = l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_18_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_457 = torch.nn.functional.relu(x_456, inplace=True)
        x_456 = None
        out_114 = torch.cat((x_448, x_457), 1)
        x_448 = x_457 = None
        out_115 = torch.conv2d(
            out_114,
            l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_114 = (
            l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_18_modules_conv2d_parameters_bias_ = None
        mul_28 = out_115 * 0.1
        out_115 = None
        out_116 = mul_28 + out_113
        mul_28 = out_113 = None
        out_117 = torch.nn.functional.relu(out_116, inplace=False)
        out_116 = None
        x_458 = torch.conv2d(
            out_117,
            l_self_modules_repeat_1_modules_19_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_19_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_459 = torch.nn.functional.batch_norm(
            x_458,
            l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_458 = l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_19_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_460 = torch.nn.functional.relu(x_459, inplace=True)
        x_459 = None
        x_461 = torch.conv2d(
            out_117,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_462 = torch.nn.functional.batch_norm(
            x_461,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_461 = l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_463 = torch.nn.functional.relu(x_462, inplace=True)
        x_462 = None
        x_464 = torch.conv2d(
            x_463,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_463 = l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_465 = torch.nn.functional.batch_norm(
            x_464,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_464 = l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_466 = torch.nn.functional.relu(x_465, inplace=True)
        x_465 = None
        x_467 = torch.conv2d(
            x_466,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_466 = l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_468 = torch.nn.functional.batch_norm(
            x_467,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_467 = l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_1_modules_19_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_469 = torch.nn.functional.relu(x_468, inplace=True)
        x_468 = None
        out_118 = torch.cat((x_460, x_469), 1)
        x_460 = x_469 = None
        out_119 = torch.conv2d(
            out_118,
            l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_118 = (
            l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_1_modules_19_modules_conv2d_parameters_bias_ = None
        mul_29 = out_119 * 0.1
        out_119 = None
        out_120 = mul_29 + out_117
        mul_29 = out_117 = None
        out_121 = torch.nn.functional.relu(out_120, inplace=False)
        out_120 = None
        x_470 = torch.conv2d(
            out_121,
            l_self_modules_mixed_7a_modules_branch0_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7a_modules_branch0_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_471 = torch.nn.functional.batch_norm(
            x_470,
            l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_470 = l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch0_modules_0_modules_bn_parameters_bias_ = (None)
        x_472 = torch.nn.functional.relu(x_471, inplace=True)
        x_471 = None
        x_473 = torch.conv2d(
            x_472,
            l_self_modules_mixed_7a_modules_branch0_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_472 = l_self_modules_mixed_7a_modules_branch0_modules_1_modules_conv_parameters_weight_ = (None)
        x_474 = torch.nn.functional.batch_norm(
            x_473,
            l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_473 = l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch0_modules_1_modules_bn_parameters_bias_ = (None)
        x_475 = torch.nn.functional.relu(x_474, inplace=True)
        x_474 = None
        x_476 = torch.conv2d(
            out_121,
            l_self_modules_mixed_7a_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7a_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_477 = torch.nn.functional.batch_norm(
            x_476,
            l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_476 = l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_478 = torch.nn.functional.relu(x_477, inplace=True)
        x_477 = None
        x_479 = torch.conv2d(
            x_478,
            l_self_modules_mixed_7a_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_478 = l_self_modules_mixed_7a_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_480 = torch.nn.functional.batch_norm(
            x_479,
            l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_479 = l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_481 = torch.nn.functional.relu(x_480, inplace=True)
        x_480 = None
        x_482 = torch.conv2d(
            out_121,
            l_self_modules_mixed_7a_modules_branch2_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7a_modules_branch2_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_483 = torch.nn.functional.batch_norm(
            x_482,
            l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_482 = l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch2_modules_0_modules_bn_parameters_bias_ = (None)
        x_484 = torch.nn.functional.relu(x_483, inplace=True)
        x_483 = None
        x_485 = torch.conv2d(
            x_484,
            l_self_modules_mixed_7a_modules_branch2_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_484 = l_self_modules_mixed_7a_modules_branch2_modules_1_modules_conv_parameters_weight_ = (None)
        x_486 = torch.nn.functional.batch_norm(
            x_485,
            l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_485 = l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch2_modules_1_modules_bn_parameters_bias_ = (None)
        x_487 = torch.nn.functional.relu(x_486, inplace=True)
        x_486 = None
        x_488 = torch.conv2d(
            x_487,
            l_self_modules_mixed_7a_modules_branch2_modules_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_487 = l_self_modules_mixed_7a_modules_branch2_modules_2_modules_conv_parameters_weight_ = (None)
        x_489 = torch.nn.functional.batch_norm(
            x_488,
            l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_488 = l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_buffers_running_var_ = l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_weight_ = l_self_modules_mixed_7a_modules_branch2_modules_2_modules_bn_parameters_bias_ = (None)
        x_490 = torch.nn.functional.relu(x_489, inplace=True)
        x_489 = None
        x3 = torch.nn.functional.max_pool2d(
            out_121, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        out_121 = None
        out_122 = torch.cat((x_475, x_481, x_490, x3), 1)
        x_475 = x_481 = x_490 = x3 = None
        x_491 = torch.conv2d(
            out_122,
            l_self_modules_repeat_2_modules_0_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_0_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_492 = torch.nn.functional.batch_norm(
            x_491,
            l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_491 = l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_0_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_493 = torch.nn.functional.relu(x_492, inplace=True)
        x_492 = None
        x_494 = torch.conv2d(
            out_122,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_495 = torch.nn.functional.batch_norm(
            x_494,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_494 = l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_496 = torch.nn.functional.relu(x_495, inplace=True)
        x_495 = None
        x_497 = torch.conv2d(
            x_496,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_496 = l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_498 = torch.nn.functional.batch_norm(
            x_497,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_497 = l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_499 = torch.nn.functional.relu(x_498, inplace=True)
        x_498 = None
        x_500 = torch.conv2d(
            x_499,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_499 = l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_501 = torch.nn.functional.batch_norm(
            x_500,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_500 = l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_0_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_502 = torch.nn.functional.relu(x_501, inplace=True)
        x_501 = None
        out_123 = torch.cat((x_493, x_502), 1)
        x_493 = x_502 = None
        out_124 = torch.conv2d(
            out_123,
            l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_123 = (
            l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_0_modules_conv2d_parameters_bias_ = None
        mul_30 = out_124 * 0.2
        out_124 = None
        out_125 = mul_30 + out_122
        mul_30 = out_122 = None
        out_126 = torch.nn.functional.relu(out_125, inplace=False)
        out_125 = None
        x_503 = torch.conv2d(
            out_126,
            l_self_modules_repeat_2_modules_1_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_1_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_504 = torch.nn.functional.batch_norm(
            x_503,
            l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_503 = l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_1_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_505 = torch.nn.functional.relu(x_504, inplace=True)
        x_504 = None
        x_506 = torch.conv2d(
            out_126,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_507 = torch.nn.functional.batch_norm(
            x_506,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_506 = l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_508 = torch.nn.functional.relu(x_507, inplace=True)
        x_507 = None
        x_509 = torch.conv2d(
            x_508,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_508 = l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_510 = torch.nn.functional.batch_norm(
            x_509,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_509 = l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_511 = torch.nn.functional.relu(x_510, inplace=True)
        x_510 = None
        x_512 = torch.conv2d(
            x_511,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_511 = l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_513 = torch.nn.functional.batch_norm(
            x_512,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_512 = l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_1_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_514 = torch.nn.functional.relu(x_513, inplace=True)
        x_513 = None
        out_127 = torch.cat((x_505, x_514), 1)
        x_505 = x_514 = None
        out_128 = torch.conv2d(
            out_127,
            l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_127 = (
            l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_1_modules_conv2d_parameters_bias_ = None
        mul_31 = out_128 * 0.2
        out_128 = None
        out_129 = mul_31 + out_126
        mul_31 = out_126 = None
        out_130 = torch.nn.functional.relu(out_129, inplace=False)
        out_129 = None
        x_515 = torch.conv2d(
            out_130,
            l_self_modules_repeat_2_modules_2_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_2_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_516 = torch.nn.functional.batch_norm(
            x_515,
            l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_515 = l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_2_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_517 = torch.nn.functional.relu(x_516, inplace=True)
        x_516 = None
        x_518 = torch.conv2d(
            out_130,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_519 = torch.nn.functional.batch_norm(
            x_518,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_518 = l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_520 = torch.nn.functional.relu(x_519, inplace=True)
        x_519 = None
        x_521 = torch.conv2d(
            x_520,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_520 = l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_522 = torch.nn.functional.batch_norm(
            x_521,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_521 = l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_523 = torch.nn.functional.relu(x_522, inplace=True)
        x_522 = None
        x_524 = torch.conv2d(
            x_523,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_523 = l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_525 = torch.nn.functional.batch_norm(
            x_524,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_524 = l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_2_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_526 = torch.nn.functional.relu(x_525, inplace=True)
        x_525 = None
        out_131 = torch.cat((x_517, x_526), 1)
        x_517 = x_526 = None
        out_132 = torch.conv2d(
            out_131,
            l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_131 = (
            l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_2_modules_conv2d_parameters_bias_ = None
        mul_32 = out_132 * 0.2
        out_132 = None
        out_133 = mul_32 + out_130
        mul_32 = out_130 = None
        out_134 = torch.nn.functional.relu(out_133, inplace=False)
        out_133 = None
        x_527 = torch.conv2d(
            out_134,
            l_self_modules_repeat_2_modules_3_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_3_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_528 = torch.nn.functional.batch_norm(
            x_527,
            l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_527 = l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_3_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_529 = torch.nn.functional.relu(x_528, inplace=True)
        x_528 = None
        x_530 = torch.conv2d(
            out_134,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_531 = torch.nn.functional.batch_norm(
            x_530,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_530 = l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_532 = torch.nn.functional.relu(x_531, inplace=True)
        x_531 = None
        x_533 = torch.conv2d(
            x_532,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_532 = l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_534 = torch.nn.functional.batch_norm(
            x_533,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_533 = l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_535 = torch.nn.functional.relu(x_534, inplace=True)
        x_534 = None
        x_536 = torch.conv2d(
            x_535,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_535 = l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_537 = torch.nn.functional.batch_norm(
            x_536,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_536 = l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_3_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_538 = torch.nn.functional.relu(x_537, inplace=True)
        x_537 = None
        out_135 = torch.cat((x_529, x_538), 1)
        x_529 = x_538 = None
        out_136 = torch.conv2d(
            out_135,
            l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_135 = (
            l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_3_modules_conv2d_parameters_bias_ = None
        mul_33 = out_136 * 0.2
        out_136 = None
        out_137 = mul_33 + out_134
        mul_33 = out_134 = None
        out_138 = torch.nn.functional.relu(out_137, inplace=False)
        out_137 = None
        x_539 = torch.conv2d(
            out_138,
            l_self_modules_repeat_2_modules_4_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_4_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_540 = torch.nn.functional.batch_norm(
            x_539,
            l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_539 = l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_4_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_541 = torch.nn.functional.relu(x_540, inplace=True)
        x_540 = None
        x_542 = torch.conv2d(
            out_138,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_543 = torch.nn.functional.batch_norm(
            x_542,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_542 = l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_544 = torch.nn.functional.relu(x_543, inplace=True)
        x_543 = None
        x_545 = torch.conv2d(
            x_544,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_544 = l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_546 = torch.nn.functional.batch_norm(
            x_545,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_545 = l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_547 = torch.nn.functional.relu(x_546, inplace=True)
        x_546 = None
        x_548 = torch.conv2d(
            x_547,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_547 = l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_549 = torch.nn.functional.batch_norm(
            x_548,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_548 = l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_4_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_550 = torch.nn.functional.relu(x_549, inplace=True)
        x_549 = None
        out_139 = torch.cat((x_541, x_550), 1)
        x_541 = x_550 = None
        out_140 = torch.conv2d(
            out_139,
            l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_139 = (
            l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_4_modules_conv2d_parameters_bias_ = None
        mul_34 = out_140 * 0.2
        out_140 = None
        out_141 = mul_34 + out_138
        mul_34 = out_138 = None
        out_142 = torch.nn.functional.relu(out_141, inplace=False)
        out_141 = None
        x_551 = torch.conv2d(
            out_142,
            l_self_modules_repeat_2_modules_5_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_5_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_552 = torch.nn.functional.batch_norm(
            x_551,
            l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_551 = l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_5_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_553 = torch.nn.functional.relu(x_552, inplace=True)
        x_552 = None
        x_554 = torch.conv2d(
            out_142,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_555 = torch.nn.functional.batch_norm(
            x_554,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_554 = l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_556 = torch.nn.functional.relu(x_555, inplace=True)
        x_555 = None
        x_557 = torch.conv2d(
            x_556,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_556 = l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_558 = torch.nn.functional.batch_norm(
            x_557,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_557 = l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_559 = torch.nn.functional.relu(x_558, inplace=True)
        x_558 = None
        x_560 = torch.conv2d(
            x_559,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_559 = l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_561 = torch.nn.functional.batch_norm(
            x_560,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_560 = l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_5_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_562 = torch.nn.functional.relu(x_561, inplace=True)
        x_561 = None
        out_143 = torch.cat((x_553, x_562), 1)
        x_553 = x_562 = None
        out_144 = torch.conv2d(
            out_143,
            l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_143 = (
            l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_5_modules_conv2d_parameters_bias_ = None
        mul_35 = out_144 * 0.2
        out_144 = None
        out_145 = mul_35 + out_142
        mul_35 = out_142 = None
        out_146 = torch.nn.functional.relu(out_145, inplace=False)
        out_145 = None
        x_563 = torch.conv2d(
            out_146,
            l_self_modules_repeat_2_modules_6_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_6_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_564 = torch.nn.functional.batch_norm(
            x_563,
            l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_563 = l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_6_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_565 = torch.nn.functional.relu(x_564, inplace=True)
        x_564 = None
        x_566 = torch.conv2d(
            out_146,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_567 = torch.nn.functional.batch_norm(
            x_566,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_566 = l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_568 = torch.nn.functional.relu(x_567, inplace=True)
        x_567 = None
        x_569 = torch.conv2d(
            x_568,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_568 = l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_570 = torch.nn.functional.batch_norm(
            x_569,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_569 = l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_571 = torch.nn.functional.relu(x_570, inplace=True)
        x_570 = None
        x_572 = torch.conv2d(
            x_571,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_571 = l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_573 = torch.nn.functional.batch_norm(
            x_572,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_572 = l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_6_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_574 = torch.nn.functional.relu(x_573, inplace=True)
        x_573 = None
        out_147 = torch.cat((x_565, x_574), 1)
        x_565 = x_574 = None
        out_148 = torch.conv2d(
            out_147,
            l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_147 = (
            l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_6_modules_conv2d_parameters_bias_ = None
        mul_36 = out_148 * 0.2
        out_148 = None
        out_149 = mul_36 + out_146
        mul_36 = out_146 = None
        out_150 = torch.nn.functional.relu(out_149, inplace=False)
        out_149 = None
        x_575 = torch.conv2d(
            out_150,
            l_self_modules_repeat_2_modules_7_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_7_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_576 = torch.nn.functional.batch_norm(
            x_575,
            l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_575 = l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_7_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_577 = torch.nn.functional.relu(x_576, inplace=True)
        x_576 = None
        x_578 = torch.conv2d(
            out_150,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_579 = torch.nn.functional.batch_norm(
            x_578,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_578 = l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_580 = torch.nn.functional.relu(x_579, inplace=True)
        x_579 = None
        x_581 = torch.conv2d(
            x_580,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_580 = l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_582 = torch.nn.functional.batch_norm(
            x_581,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_581 = l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_583 = torch.nn.functional.relu(x_582, inplace=True)
        x_582 = None
        x_584 = torch.conv2d(
            x_583,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_583 = l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_585 = torch.nn.functional.batch_norm(
            x_584,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_584 = l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_7_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_586 = torch.nn.functional.relu(x_585, inplace=True)
        x_585 = None
        out_151 = torch.cat((x_577, x_586), 1)
        x_577 = x_586 = None
        out_152 = torch.conv2d(
            out_151,
            l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_151 = (
            l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_7_modules_conv2d_parameters_bias_ = None
        mul_37 = out_152 * 0.2
        out_152 = None
        out_153 = mul_37 + out_150
        mul_37 = out_150 = None
        out_154 = torch.nn.functional.relu(out_153, inplace=False)
        out_153 = None
        x_587 = torch.conv2d(
            out_154,
            l_self_modules_repeat_2_modules_8_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_8_modules_branch0_modules_conv_parameters_weight_ = (
            None
        )
        x_588 = torch.nn.functional.batch_norm(
            x_587,
            l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_587 = l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_8_modules_branch0_modules_bn_parameters_bias_ = (None)
        x_589 = torch.nn.functional.relu(x_588, inplace=True)
        x_588 = None
        x_590 = torch.conv2d(
            out_154,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_591 = torch.nn.functional.batch_norm(
            x_590,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_590 = l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_0_modules_bn_parameters_bias_ = (None)
        x_592 = torch.nn.functional.relu(x_591, inplace=True)
        x_591 = None
        x_593 = torch.conv2d(
            x_592,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_592 = l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_594 = torch.nn.functional.batch_norm(
            x_593,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_593 = l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_1_modules_bn_parameters_bias_ = (None)
        x_595 = torch.nn.functional.relu(x_594, inplace=True)
        x_594 = None
        x_596 = torch.conv2d(
            x_595,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_595 = l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_597 = torch.nn.functional.batch_norm(
            x_596,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_596 = l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_weight_ = l_self_modules_repeat_2_modules_8_modules_branch1_modules_2_modules_bn_parameters_bias_ = (None)
        x_598 = torch.nn.functional.relu(x_597, inplace=True)
        x_597 = None
        out_155 = torch.cat((x_589, x_598), 1)
        x_589 = x_598 = None
        out_156 = torch.conv2d(
            out_155,
            l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_weight_,
            l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_155 = (
            l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_weight_
        ) = l_self_modules_repeat_2_modules_8_modules_conv2d_parameters_bias_ = None
        mul_38 = out_156 * 0.2
        out_156 = None
        out_157 = mul_38 + out_154
        mul_38 = out_154 = None
        out_158 = torch.nn.functional.relu(out_157, inplace=False)
        out_157 = None
        x_599 = torch.conv2d(
            out_158,
            l_self_modules_block8_modules_branch0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_block8_modules_branch0_modules_conv_parameters_weight_ = None
        x_600 = torch.nn.functional.batch_norm(
            x_599,
            l_self_modules_block8_modules_branch0_modules_bn_buffers_running_mean_,
            l_self_modules_block8_modules_branch0_modules_bn_buffers_running_var_,
            l_self_modules_block8_modules_branch0_modules_bn_parameters_weight_,
            l_self_modules_block8_modules_branch0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_599 = (
            l_self_modules_block8_modules_branch0_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_block8_modules_branch0_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_block8_modules_branch0_modules_bn_parameters_weight_
        ) = l_self_modules_block8_modules_branch0_modules_bn_parameters_bias_ = None
        x_601 = torch.nn.functional.relu(x_600, inplace=True)
        x_600 = None
        x_602 = torch.conv2d(
            out_158,
            l_self_modules_block8_modules_branch1_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_block8_modules_branch1_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_603 = torch.nn.functional.batch_norm(
            x_602,
            l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_weight_,
            l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_602 = l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_block8_modules_branch1_modules_0_modules_bn_buffers_running_var_ = l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_weight_ = (
            l_self_modules_block8_modules_branch1_modules_0_modules_bn_parameters_bias_
        ) = None
        x_604 = torch.nn.functional.relu(x_603, inplace=True)
        x_603 = None
        x_605 = torch.conv2d(
            x_604,
            l_self_modules_block8_modules_branch1_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        x_604 = l_self_modules_block8_modules_branch1_modules_1_modules_conv_parameters_weight_ = (None)
        x_606 = torch.nn.functional.batch_norm(
            x_605,
            l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_weight_,
            l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_605 = l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_block8_modules_branch1_modules_1_modules_bn_buffers_running_var_ = l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_weight_ = (
            l_self_modules_block8_modules_branch1_modules_1_modules_bn_parameters_bias_
        ) = None
        x_607 = torch.nn.functional.relu(x_606, inplace=True)
        x_606 = None
        x_608 = torch.conv2d(
            x_607,
            l_self_modules_block8_modules_branch1_modules_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_607 = l_self_modules_block8_modules_branch1_modules_2_modules_conv_parameters_weight_ = (None)
        x_609 = torch.nn.functional.batch_norm(
            x_608,
            l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_mean_,
            l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_var_,
            l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_weight_,
            l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_608 = l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_mean_ = l_self_modules_block8_modules_branch1_modules_2_modules_bn_buffers_running_var_ = l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_weight_ = (
            l_self_modules_block8_modules_branch1_modules_2_modules_bn_parameters_bias_
        ) = None
        x_610 = torch.nn.functional.relu(x_609, inplace=True)
        x_609 = None
        out_159 = torch.cat((x_601, x_610), 1)
        x_601 = x_610 = None
        out_160 = torch.conv2d(
            out_159,
            l_self_modules_block8_modules_conv2d_parameters_weight_,
            l_self_modules_block8_modules_conv2d_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_159 = (
            l_self_modules_block8_modules_conv2d_parameters_weight_
        ) = l_self_modules_block8_modules_conv2d_parameters_bias_ = None
        mul_39 = out_160 * 1.0
        out_160 = None
        out_161 = mul_39 + out_158
        mul_39 = out_158 = None
        x_611 = torch.conv2d(
            out_161,
            l_self_modules_conv2d_7b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_161 = l_self_modules_conv2d_7b_modules_conv_parameters_weight_ = None
        x_612 = torch.nn.functional.batch_norm(
            x_611,
            l_self_modules_conv2d_7b_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_7b_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_7b_modules_bn_parameters_weight_,
            l_self_modules_conv2d_7b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_611 = (
            l_self_modules_conv2d_7b_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_7b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_7b_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_7b_modules_bn_parameters_bias_ = None
        x_613 = torch.nn.functional.relu(x_612, inplace=True)
        x_612 = None
        x_614 = torch.nn.functional.adaptive_avg_pool2d(x_613, 1)
        x_613 = None
        x_615 = x_614.flatten(1, -1)
        x_614 = None
        x_616 = torch.nn.functional.dropout(x_615, 0.0, False, False)
        x_615 = None
        x_617 = torch._C._nn.linear(
            x_616,
            l_self_modules_classif_parameters_weight_,
            l_self_modules_classif_parameters_bias_,
        )
        x_616 = (
            l_self_modules_classif_parameters_weight_
        ) = l_self_modules_classif_parameters_bias_ = None
        return (x_617,)
