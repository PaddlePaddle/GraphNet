import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_Conv2d_1a_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2a_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2b_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_3b_1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_4a_3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch1x1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch_pool_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv2d_1a_3x3_modules_conv_parameters_weight_ = (
            L_self_modules_Conv2d_1a_3x3_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_var_ = (
            L_self_modules_Conv2d_1a_3x3_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_1a_3x3_modules_bn_parameters_weight_ = (
            L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_1a_3x3_modules_bn_parameters_bias_ = (
            L_self_modules_Conv2d_1a_3x3_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_2a_3x3_modules_conv_parameters_weight_ = (
            L_self_modules_Conv2d_2a_3x3_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_var_ = (
            L_self_modules_Conv2d_2a_3x3_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_2a_3x3_modules_bn_parameters_weight_ = (
            L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_2a_3x3_modules_bn_parameters_bias_ = (
            L_self_modules_Conv2d_2a_3x3_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_2b_3x3_modules_conv_parameters_weight_ = (
            L_self_modules_Conv2d_2b_3x3_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_var_ = (
            L_self_modules_Conv2d_2b_3x3_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_2b_3x3_modules_bn_parameters_weight_ = (
            L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_2b_3x3_modules_bn_parameters_bias_ = (
            L_self_modules_Conv2d_2b_3x3_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_3b_1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Conv2d_3b_1x1_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Conv2d_3b_1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_3b_1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_3b_1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Conv2d_3b_1x1_modules_bn_parameters_bias_
        )
        l_self_modules_conv2d_4a_3x3_modules_conv_parameters_weight_ = (
            L_self_modules_Conv2d_4a_3x3_modules_conv_parameters_weight_
        )
        l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_mean_
        )
        l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_var_ = (
            L_self_modules_Conv2d_4a_3x3_modules_bn_buffers_running_var_
        )
        l_self_modules_conv2d_4a_3x3_modules_bn_parameters_weight_ = (
            L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_weight_
        )
        l_self_modules_conv2d_4a_3x3_modules_bn_parameters_bias_ = (
            L_self_modules_Conv2d_4a_3x3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5b_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5b_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5c_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5c_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_5d_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_5d_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6a_modules_branch3x3_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6a_modules_branch3x3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_
        l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_
        l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_
        l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_
        l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6b_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6b_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_
        l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_
        l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_
        l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_
        l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6c_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6c_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_
        l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_
        l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_
        l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_
        l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6d_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6d_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_
        l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_
        l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_
        l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_
        l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_
        l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_
        l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_6e_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_6e_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_ = L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_ = L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_
        l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_
        l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_ = L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_
        l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7b_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7b_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch1x1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch1x1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_
        l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_ = L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_
        l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_
        )
        l_self_modules_mixed_7c_modules_branch_pool_modules_conv_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch_pool_modules_conv_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_ = (
            L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_
        )
        l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_ = (
            L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_
        )
        l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_weight_ = (
            L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_weight_
        )
        l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_bias_ = (
            L_self_modules_Mixed_7c_modules_branch_pool_modules_bn_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv2d_1a_3x3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv2d_1a_3x3_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_1a_3x3_modules_bn_parameters_weight_,
            l_self_modules_conv2d_1a_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x = (
            l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_1a_3x3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_1a_3x3_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_1a_3x3_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_conv2d_2a_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_conv2d_2a_3x3_modules_conv_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_2a_3x3_modules_bn_parameters_weight_,
            l_self_modules_conv2d_2a_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_3 = (
            l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_2a_3x3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_2a_3x3_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_2a_3x3_modules_bn_parameters_bias_ = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_conv2d_2b_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_5 = l_self_modules_conv2d_2b_3x3_modules_conv_parameters_weight_ = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_2b_3x3_modules_bn_parameters_weight_,
            l_self_modules_conv2d_2b_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_6 = (
            l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_2b_3x3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_2b_3x3_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_2b_3x3_modules_bn_parameters_bias_ = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.nn.functional.max_pool2d(
            x_8, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_conv2d_3b_1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_conv2d_3b_1x1_modules_conv_parameters_weight_ = None
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_3b_1x1_modules_bn_parameters_weight_,
            l_self_modules_conv2d_3b_1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_10 = (
            l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_3b_1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_3b_1x1_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_3b_1x1_modules_bn_parameters_bias_ = None
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        x_13 = torch.conv2d(
            x_12,
            l_self_modules_conv2d_4a_3x3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_12 = l_self_modules_conv2d_4a_3x3_modules_conv_parameters_weight_ = None
        x_14 = torch.nn.functional.batch_norm(
            x_13,
            l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_mean_,
            l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_var_,
            l_self_modules_conv2d_4a_3x3_modules_bn_parameters_weight_,
            l_self_modules_conv2d_4a_3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_13 = (
            l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_conv2d_4a_3x3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_conv2d_4a_3x3_modules_bn_parameters_weight_
        ) = l_self_modules_conv2d_4a_3x3_modules_bn_parameters_bias_ = None
        x_15 = torch.nn.functional.relu(x_14, inplace=True)
        x_14 = None
        x_16 = torch.nn.functional.max_pool2d(
            x_15, 3, 2, 0, 1, ceil_mode=False, return_indices=False
        )
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_18 = torch.nn.functional.batch_norm(
            x_17,
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_17 = (
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_5b_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            None
        )
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_20 = (
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_1_modules_bn_parameters_bias_
        ) = None
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        x_22 = (
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_conv_parameters_weight_
        ) = None
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_23 = (
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch5x5_2_modules_bn_parameters_bias_
        ) = None
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_16,
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_27 = torch.nn.functional.batch_norm(
            x_26,
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_26 = l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_29 = l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_32 = l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        ) = None
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        branch_pool = torch._C._nn.avg_pool2d(x_16, kernel_size=3, stride=1, padding=1)
        x_16 = None
        x_35 = torch.conv2d(
            branch_pool,
            l_self_modules_mixed_5b_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool = (
            l_self_modules_mixed_5b_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_35 = (
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5b_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.cat([x_19, x_25, x_34, x_37], 1)
        x_19 = x_25 = x_34 = x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_mixed_5c_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5c_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_39 = (
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_5c_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_38,
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5c_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            None
        )
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_42 = (
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_1_modules_bn_parameters_bias_
        ) = None
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        x_44 = (
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_conv_parameters_weight_
        ) = None
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_45 = (
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch5x5_2_modules_bn_parameters_bias_
        ) = None
        x_47 = torch.nn.functional.relu(x_46, inplace=True)
        x_46 = None
        x_48 = torch.conv2d(
            x_38,
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_49 = torch.nn.functional.batch_norm(
            x_48,
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_48 = l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_50 = torch.nn.functional.relu(x_49, inplace=True)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_50 = l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_51 = l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_54 = l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        branch_pool_1 = torch._C._nn.avg_pool2d(
            x_38, kernel_size=3, stride=1, padding=1
        )
        x_38 = None
        x_57 = torch.conv2d(
            branch_pool_1,
            l_self_modules_mixed_5c_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_1 = (
            l_self_modules_mixed_5c_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_57 = (
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5c_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.cat([x_41, x_47, x_56, x_59], 1)
        x_41 = x_47 = x_56 = x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_mixed_5d_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5d_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_61 = (
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_5d_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        x_64 = torch.conv2d(
            x_60,
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5d_modules_branch5x5_1_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.nn.functional.batch_norm(
            x_64,
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_64 = (
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_1_modules_bn_parameters_bias_
        ) = None
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (1, 1),
            1,
        )
        x_66 = (
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_conv_parameters_weight_
        ) = None
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_67 = (
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch5x5_2_modules_bn_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_60,
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_70 = l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_72 = l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_73 = l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_76 = l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        branch_pool_2 = torch._C._nn.avg_pool2d(
            x_60, kernel_size=3, stride=1, padding=1
        )
        x_60 = None
        x_79 = torch.conv2d(
            branch_pool_2,
            l_self_modules_mixed_5d_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_2 = (
            l_self_modules_mixed_5d_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_79 = (
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_5d_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.cat([x_63, x_69, x_78, x_81], 1)
        x_63 = x_69 = x_78 = x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_mixed_6a_modules_branch3x3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6a_modules_branch3x3_modules_conv_parameters_weight_ = None
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_83 = (
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6a_modules_branch3x3_modules_bn_parameters_bias_ = None
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_82,
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_86 = l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_89 = l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_conv_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_92 = l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6a_modules_branch3x3dbl_3_modules_bn_parameters_bias_
        ) = None
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        branch_pool_3 = torch.nn.functional.max_pool2d(x_82, kernel_size=3, stride=2)
        x_82 = None
        x_95 = torch.cat([x_85, x_94, branch_pool_3], 1)
        x_85 = x_94 = branch_pool_3 = None
        x_96 = torch.conv2d(
            x_95,
            l_self_modules_mixed_6b_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6b_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_97 = torch.nn.functional.batch_norm(
            x_96,
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_96 = (
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6b_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_98 = torch.nn.functional.relu(x_97, inplace=True)
        x_97 = None
        x_99 = torch.conv2d(
            x_95,
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6b_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_99 = (
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_1_modules_bn_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_101 = (
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_conv_parameters_weight_
        ) = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_102 = (
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_2_modules_bn_parameters_bias_
        ) = None
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_104 = (
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_conv_parameters_weight_
        ) = None
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_105 = (
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7_3_modules_bn_parameters_bias_
        ) = None
        x_107 = torch.nn.functional.relu(x_106, inplace=True)
        x_106 = None
        x_108 = torch.conv2d(
            x_95,
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_108 = l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        ) = None
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_110 = l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = (None)
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_111 = l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_113 = l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = (None)
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_114 = l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_116 = l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = (None)
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_117 = l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        ) = None
        x_119 = torch.nn.functional.relu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = (None)
        x_121 = torch.nn.functional.batch_norm(
            x_120,
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_120 = l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        ) = None
        x_122 = torch.nn.functional.relu(x_121, inplace=True)
        x_121 = None
        branch_pool_4 = torch._C._nn.avg_pool2d(
            x_95, kernel_size=3, stride=1, padding=1
        )
        x_95 = None
        x_123 = torch.conv2d(
            branch_pool_4,
            l_self_modules_mixed_6b_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_4 = (
            l_self_modules_mixed_6b_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_124 = torch.nn.functional.batch_norm(
            x_123,
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_123 = (
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6b_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_125 = torch.nn.functional.relu(x_124, inplace=True)
        x_124 = None
        x_126 = torch.cat([x_98, x_107, x_122, x_125], 1)
        x_98 = x_107 = x_122 = x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_mixed_6c_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6c_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_127 = (
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6c_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_126,
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6c_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_130 = (
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_1_modules_bn_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_132 = (
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_conv_parameters_weight_
        ) = None
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_133 = (
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_2_modules_bn_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        x_136 = torch.conv2d(
            x_135,
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_135 = (
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_conv_parameters_weight_
        ) = None
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_136 = (
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7_3_modules_bn_parameters_bias_
        ) = None
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        x_139 = torch.conv2d(
            x_126,
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_139 = l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        ) = None
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = (None)
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_142 = l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        ) = None
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_144 = l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = (None)
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_145 = l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        ) = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_147 = l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_148 = l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        ) = None
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_151 = l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        branch_pool_5 = torch._C._nn.avg_pool2d(
            x_126, kernel_size=3, stride=1, padding=1
        )
        x_126 = None
        x_154 = torch.conv2d(
            branch_pool_5,
            l_self_modules_mixed_6c_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_5 = (
            l_self_modules_mixed_6c_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_154 = (
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6c_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.cat([x_129, x_138, x_153, x_156], 1)
        x_129 = x_138 = x_153 = x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_mixed_6d_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6d_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_159 = torch.nn.functional.batch_norm(
            x_158,
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_158 = (
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6d_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_160 = torch.nn.functional.relu(x_159, inplace=True)
        x_159 = None
        x_161 = torch.conv2d(
            x_157,
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6d_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            None
        )
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_161 = (
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_1_modules_bn_parameters_bias_
        ) = None
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_163 = (
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_conv_parameters_weight_
        ) = None
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_164 = (
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_2_modules_bn_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_166 = (
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_conv_parameters_weight_
        ) = None
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_167 = (
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7_3_modules_bn_parameters_bias_
        ) = None
        x_169 = torch.nn.functional.relu(x_168, inplace=True)
        x_168 = None
        x_170 = torch.conv2d(
            x_157,
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_170 = l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        ) = None
        x_172 = torch.nn.functional.relu(x_171, inplace=True)
        x_171 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_173 = l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        ) = None
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = (None)
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_176 = l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        ) = None
        x_178 = torch.nn.functional.relu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_178 = l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = (None)
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_179 = l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        ) = None
        x_181 = torch.nn.functional.relu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_182 = l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        ) = None
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        branch_pool_6 = torch._C._nn.avg_pool2d(
            x_157, kernel_size=3, stride=1, padding=1
        )
        x_157 = None
        x_185 = torch.conv2d(
            branch_pool_6,
            l_self_modules_mixed_6d_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_6 = (
            l_self_modules_mixed_6d_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_185 = (
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6d_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_187 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        x_188 = torch.cat([x_160, x_169, x_184, x_187], 1)
        x_160 = x_169 = x_184 = x_187 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_mixed_6e_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6e_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_189 = (
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_6e_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_191 = torch.nn.functional.relu(x_190, inplace=True)
        x_190 = None
        x_192 = torch.conv2d(
            x_188,
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6e_modules_branch7x7_1_modules_conv_parameters_weight_ = (
            None
        )
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_192 = (
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_1_modules_bn_parameters_bias_
        ) = None
        x_194 = torch.nn.functional.relu(x_193, inplace=True)
        x_193 = None
        x_195 = torch.conv2d(
            x_194,
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_194 = (
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_conv_parameters_weight_
        ) = None
        x_196 = torch.nn.functional.batch_norm(
            x_195,
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_195 = (
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_2_modules_bn_parameters_bias_
        ) = None
        x_197 = torch.nn.functional.relu(x_196, inplace=True)
        x_196 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_197 = (
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_conv_parameters_weight_
        ) = None
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_198 = (
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7_3_modules_bn_parameters_bias_
        ) = None
        x_200 = torch.nn.functional.relu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_188,
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_201 = l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_1_modules_bn_parameters_bias_
        ) = None
        x_203 = torch.nn.functional.relu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_203 = l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_conv_parameters_weight_ = (None)
        x_205 = torch.nn.functional.batch_norm(
            x_204,
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_204 = l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_2_modules_bn_parameters_bias_
        ) = None
        x_206 = torch.nn.functional.relu(x_205, inplace=True)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_206 = l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_conv_parameters_weight_ = (None)
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_207 = l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_3_modules_bn_parameters_bias_
        ) = None
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_210 = l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_4_modules_bn_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_213 = torch.conv2d(
            x_212,
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_212 = l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_conv_parameters_weight_ = (None)
        x_214 = torch.nn.functional.batch_norm(
            x_213,
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_213 = l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_mean_ = l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch7x7dbl_5_modules_bn_parameters_bias_
        ) = None
        x_215 = torch.nn.functional.relu(x_214, inplace=True)
        x_214 = None
        branch_pool_7 = torch._C._nn.avg_pool2d(
            x_188, kernel_size=3, stride=1, padding=1
        )
        x_188 = None
        x_216 = torch.conv2d(
            branch_pool_7,
            l_self_modules_mixed_6e_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_7 = (
            l_self_modules_mixed_6e_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_216 = (
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_6e_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_218 = torch.nn.functional.relu(x_217, inplace=True)
        x_217 = None
        x_219 = torch.cat([x_191, x_200, x_215, x_218], 1)
        x_191 = x_200 = x_215 = x_218 = None
        x_220 = torch.conv2d(
            x_219,
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7a_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            None
        )
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_220 = (
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_1_modules_bn_parameters_bias_
        ) = None
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_222 = (
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_conv_parameters_weight_
        ) = None
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_223 = (
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch3x3_2_modules_bn_parameters_bias_
        ) = None
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_219,
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_conv_parameters_weight_ = (
            None
        )
        x_227 = torch.nn.functional.batch_norm(
            x_226,
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_226 = l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch7x7x3_1_modules_bn_parameters_bias_
        ) = None
        x_228 = torch.nn.functional.relu(x_227, inplace=True)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 3),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_conv_parameters_weight_ = (None)
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_229 = l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch7x7x3_2_modules_bn_parameters_bias_
        ) = None
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (3, 0),
            (1, 1),
            1,
        )
        x_231 = l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_conv_parameters_weight_ = (None)
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_232 = l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch7x7x3_3_modules_bn_parameters_bias_
        ) = None
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_234 = l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_conv_parameters_weight_ = (None)
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_,
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_235 = l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7a_modules_branch7x7x3_4_modules_bn_parameters_bias_
        ) = None
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        branch_pool_8 = torch.nn.functional.max_pool2d(x_219, kernel_size=3, stride=2)
        x_219 = None
        x_238 = torch.cat([x_225, x_237, branch_pool_8], 1)
        x_225 = x_237 = branch_pool_8 = None
        x_239 = torch.conv2d(
            x_238,
            l_self_modules_mixed_7b_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7b_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_240 = torch.nn.functional.batch_norm(
            x_239,
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_239 = (
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_7b_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_241 = torch.nn.functional.relu(x_240, inplace=True)
        x_240 = None
        x_242 = torch.conv2d(
            x_238,
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7b_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            None
        )
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_242 = (
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_1_modules_bn_parameters_bias_
        ) = None
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7b_modules_branch3x3_2a_modules_conv_parameters_weight_ = (
            None
        )
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_245 = l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_mean_ = (
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_2a_modules_bn_parameters_bias_
        ) = None
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_244,
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_244 = (
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_conv_parameters_weight_
        ) = None
        x_249 = torch.nn.functional.batch_norm(
            x_248,
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_248 = l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_mean_ = (
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3_2b_modules_bn_parameters_bias_
        ) = None
        x_250 = torch.nn.functional.relu(x_249, inplace=True)
        x_249 = None
        branch3x3 = torch.cat([x_247, x_250], 1)
        x_247 = x_250 = None
        x_251 = torch.conv2d(
            x_238,
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_252 = torch.nn.functional.batch_norm(
            x_251,
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_251 = l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_253 = torch.nn.functional.relu(x_252, inplace=True)
        x_252 = None
        x_254 = torch.conv2d(
            x_253,
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_253 = l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_255 = torch.nn.functional.batch_norm(
            x_254,
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_254 = l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_256 = torch.nn.functional.relu(x_255, inplace=True)
        x_255 = None
        x_257 = torch.conv2d(
            x_256,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_conv_parameters_weight_ = (
            None
        )
        x_258 = torch.nn.functional.batch_norm(
            x_257,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_257 = l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_ = l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_weight_ = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_3a_modules_bn_parameters_bias_
        ) = None
        x_259 = torch.nn.functional.relu(x_258, inplace=True)
        x_258 = None
        x_260 = torch.conv2d(
            x_256,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_256 = l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_conv_parameters_weight_ = (None)
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_260 = l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_ = l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_weight_ = (
            l_self_modules_mixed_7b_modules_branch3x3dbl_3b_modules_bn_parameters_bias_
        ) = None
        x_262 = torch.nn.functional.relu(x_261, inplace=True)
        x_261 = None
        branch3x3dbl = torch.cat([x_259, x_262], 1)
        x_259 = x_262 = None
        branch_pool_9 = torch._C._nn.avg_pool2d(
            x_238, kernel_size=3, stride=1, padding=1
        )
        x_238 = None
        x_263 = torch.conv2d(
            branch_pool_9,
            l_self_modules_mixed_7b_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_9 = (
            l_self_modules_mixed_7b_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_263 = (
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7b_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_266 = torch.cat([x_241, branch3x3, branch3x3dbl, x_265], 1)
        x_241 = branch3x3 = branch3x3dbl = x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_mixed_7c_modules_branch1x1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7c_modules_branch1x1_modules_conv_parameters_weight_ = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_267 = (
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_weight_
        ) = l_self_modules_mixed_7c_modules_branch1x1_modules_bn_parameters_bias_ = None
        x_269 = torch.nn.functional.relu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_266,
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7c_modules_branch3x3_1_modules_conv_parameters_weight_ = (
            None
        )
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_270 = (
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_1_modules_bn_parameters_bias_
        ) = None
        x_272 = torch.nn.functional.relu(x_271, inplace=True)
        x_271 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7c_modules_branch3x3_2a_modules_conv_parameters_weight_ = (
            None
        )
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_273 = l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_mean_ = (
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_2a_modules_bn_parameters_bias_
        ) = None
        x_275 = torch.nn.functional.relu(x_274, inplace=True)
        x_274 = None
        x_276 = torch.conv2d(
            x_272,
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_272 = (
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_conv_parameters_weight_
        ) = None
        x_277 = torch.nn.functional.batch_norm(
            x_276,
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_276 = l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_mean_ = (
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3_2b_modules_bn_parameters_bias_
        ) = None
        x_278 = torch.nn.functional.relu(x_277, inplace=True)
        x_277 = None
        branch3x3_1 = torch.cat([x_275, x_278], 1)
        x_275 = x_278 = None
        x_279 = torch.conv2d(
            x_266,
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_conv_parameters_weight_ = (
            None
        )
        x_280 = torch.nn.functional.batch_norm(
            x_279,
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_279 = l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_1_modules_bn_parameters_bias_
        ) = None
        x_281 = torch.nn.functional.relu(x_280, inplace=True)
        x_280 = None
        x_282 = torch.conv2d(
            x_281,
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_281 = l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_conv_parameters_weight_ = (None)
        x_283 = torch.nn.functional.batch_norm(
            x_282,
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_282 = l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_buffers_running_var_ = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_2_modules_bn_parameters_bias_
        ) = None
        x_284 = torch.nn.functional.relu(x_283, inplace=True)
        x_283 = None
        x_285 = torch.conv2d(
            x_284,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 1),
            (1, 1),
            1,
        )
        l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_conv_parameters_weight_ = (
            None
        )
        x_286 = torch.nn.functional.batch_norm(
            x_285,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_285 = l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_buffers_running_var_ = l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_weight_ = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_3a_modules_bn_parameters_bias_
        ) = None
        x_287 = torch.nn.functional.relu(x_286, inplace=True)
        x_286 = None
        x_288 = torch.conv2d(
            x_284,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 0),
            (1, 1),
            1,
        )
        x_284 = l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_conv_parameters_weight_ = (None)
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_288 = l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_mean_ = l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_buffers_running_var_ = l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_weight_ = (
            l_self_modules_mixed_7c_modules_branch3x3dbl_3b_modules_bn_parameters_bias_
        ) = None
        x_290 = torch.nn.functional.relu(x_289, inplace=True)
        x_289 = None
        branch3x3dbl_1 = torch.cat([x_287, x_290], 1)
        x_287 = x_290 = None
        branch_pool_10 = torch._C._nn.avg_pool2d(
            x_266, kernel_size=3, stride=1, padding=1
        )
        x_266 = None
        x_291 = torch.conv2d(
            branch_pool_10,
            l_self_modules_mixed_7c_modules_branch_pool_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        branch_pool_10 = (
            l_self_modules_mixed_7c_modules_branch_pool_modules_conv_parameters_weight_
        ) = None
        x_292 = torch.nn.functional.batch_norm(
            x_291,
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_,
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_,
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_weight_,
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_bias_,
            False,
            0.1,
            0.001,
        )
        x_291 = (
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_weight_
        ) = (
            l_self_modules_mixed_7c_modules_branch_pool_modules_bn_parameters_bias_
        ) = None
        x_293 = torch.nn.functional.relu(x_292, inplace=True)
        x_292 = None
        x_294 = torch.cat([x_269, branch3x3_1, branch3x3dbl_1, x_293], 1)
        x_269 = branch3x3_1 = branch3x3dbl_1 = x_293 = None
        x_295 = torch.nn.functional.adaptive_avg_pool2d(x_294, 1)
        x_294 = None
        x_296 = x_295.flatten(1, -1)
        x_295 = None
        x_297 = torch.nn.functional.dropout(x_296, 0.0, False, False)
        x_296 = None
        x_298 = torch._C._nn.linear(
            x_297,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_297 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_298,)
