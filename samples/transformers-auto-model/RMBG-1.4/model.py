import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_x_: torch.Tensor,
        L_self_modules_conv_in_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv_in_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_: torch.Tensor,
        L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_side6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_side6_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_x_ = L_x_
        l_self_modules_conv_in_parameters_weight_ = (
            L_self_modules_conv_in_parameters_weight_
        )
        l_self_modules_conv_in_parameters_bias_ = (
            L_self_modules_conv_in_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        )
        l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_ = (
            L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        )
        l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_ = (
            L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        )
        l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_ = (
            L_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        )
        l_self_modules_side1_parameters_weight_ = (
            L_self_modules_side1_parameters_weight_
        )
        l_self_modules_side1_parameters_bias_ = L_self_modules_side1_parameters_bias_
        l_self_modules_side2_parameters_weight_ = (
            L_self_modules_side2_parameters_weight_
        )
        l_self_modules_side2_parameters_bias_ = L_self_modules_side2_parameters_bias_
        l_self_modules_side3_parameters_weight_ = (
            L_self_modules_side3_parameters_weight_
        )
        l_self_modules_side3_parameters_bias_ = L_self_modules_side3_parameters_bias_
        l_self_modules_side4_parameters_weight_ = (
            L_self_modules_side4_parameters_weight_
        )
        l_self_modules_side4_parameters_bias_ = L_self_modules_side4_parameters_bias_
        l_self_modules_side5_parameters_weight_ = (
            L_self_modules_side5_parameters_weight_
        )
        l_self_modules_side5_parameters_bias_ = L_self_modules_side5_parameters_bias_
        l_self_modules_side6_parameters_weight_ = (
            L_self_modules_side6_parameters_weight_
        )
        l_self_modules_side6_parameters_bias_ = L_self_modules_side6_parameters_bias_
        hxin = torch.conv2d(
            l_x_,
            l_self_modules_conv_in_parameters_weight_,
            l_self_modules_conv_in_parameters_bias_,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = (
            l_self_modules_conv_in_parameters_weight_
        ) = l_self_modules_conv_in_parameters_bias_ = None
        conv2d_1 = torch.conv2d(
            hxin,
            l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hxin = (
            l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm = torch.nn.functional.batch_norm(
            conv2d_1,
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_1 = (
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout = torch.nn.functional.relu(batch_norm, inplace=True)
        batch_norm = None
        conv2d_2 = torch.conv2d(
            xout,
            l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage1_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_1 = torch.nn.functional.batch_norm(
            conv2d_2,
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_2 = (
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_1 = torch.nn.functional.relu(batch_norm_1, inplace=True)
        batch_norm_1 = None
        hx = torch.nn.functional.max_pool2d(
            xout_1, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_3 = torch.conv2d(
            hx,
            l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx = (
            l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_2 = torch.nn.functional.batch_norm(
            conv2d_3,
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_3 = (
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_2 = torch.nn.functional.relu(batch_norm_2, inplace=True)
        batch_norm_2 = None
        hx_1 = torch.nn.functional.max_pool2d(
            xout_2, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_4 = torch.conv2d(
            hx_1,
            l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_1 = (
            l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_3 = torch.nn.functional.batch_norm(
            conv2d_4,
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_4 = (
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_3 = torch.nn.functional.relu(batch_norm_3, inplace=True)
        batch_norm_3 = None
        hx_2 = torch.nn.functional.max_pool2d(
            xout_3, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_5 = torch.conv2d(
            hx_2,
            l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_2 = (
            l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_4 = torch.nn.functional.batch_norm(
            conv2d_5,
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_5 = (
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_4 = torch.nn.functional.relu(batch_norm_4, inplace=True)
        batch_norm_4 = None
        hx_3 = torch.nn.functional.max_pool2d(
            xout_4, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_6 = torch.conv2d(
            hx_3,
            l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_3 = (
            l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_5 = torch.nn.functional.batch_norm(
            conv2d_6,
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_6 = (
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_5 = torch.nn.functional.relu(batch_norm_5, inplace=True)
        batch_norm_5 = None
        hx_4 = torch.nn.functional.max_pool2d(
            xout_5, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_7 = torch.conv2d(
            hx_4,
            l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_4 = (
            l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv6_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_6 = torch.nn.functional.batch_norm(
            conv2d_7,
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_7 = (
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv6_modules_bn_s1_parameters_bias_
        ) = None
        xout_6 = torch.nn.functional.relu(batch_norm_6, inplace=True)
        batch_norm_6 = None
        conv2d_8 = torch.conv2d(
            xout_6,
            l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage1_modules_rebnconv7_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_7 = torch.nn.functional.batch_norm(
            conv2d_8,
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_8 = (
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv7_modules_bn_s1_parameters_bias_
        ) = None
        xout_7 = torch.nn.functional.relu(batch_norm_7, inplace=True)
        batch_norm_7 = None
        hx_5 = torch.cat((xout_7, xout_6), 1)
        xout_7 = xout_6 = None
        conv2d_9 = torch.conv2d(
            hx_5,
            l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_5 = (
            l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv6d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_8 = torch.nn.functional.batch_norm(
            conv2d_9,
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_9 = (
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv6d_modules_bn_s1_parameters_bias_
        ) = None
        xout_8 = torch.nn.functional.relu(batch_norm_8, inplace=True)
        batch_norm_8 = None
        src = torch.nn.functional.interpolate(xout_8, size=(20, 20), mode="bilinear")
        xout_8 = None
        hx_6 = torch.cat((src, xout_5), 1)
        src = xout_5 = None
        conv2d_10 = torch.conv2d(
            hx_6,
            l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_6 = (
            l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_9 = torch.nn.functional.batch_norm(
            conv2d_10,
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_10 = (
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        ) = None
        xout_9 = torch.nn.functional.relu(batch_norm_9, inplace=True)
        batch_norm_9 = None
        src_1 = torch.nn.functional.interpolate(xout_9, size=(40, 40), mode="bilinear")
        xout_9 = None
        hx_7 = torch.cat((src_1, xout_4), 1)
        src_1 = xout_4 = None
        conv2d_11 = torch.conv2d(
            hx_7,
            l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_7 = (
            l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_10 = torch.nn.functional.batch_norm(
            conv2d_11,
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_11 = (
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_10 = torch.nn.functional.relu(batch_norm_10, inplace=True)
        batch_norm_10 = None
        src_2 = torch.nn.functional.interpolate(xout_10, size=(80, 80), mode="bilinear")
        xout_10 = None
        hx_8 = torch.cat((src_2, xout_3), 1)
        src_2 = xout_3 = None
        conv2d_12 = torch.conv2d(
            hx_8,
            l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_8 = (
            l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_11 = torch.nn.functional.batch_norm(
            conv2d_12,
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_12 = (
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_11 = torch.nn.functional.relu(batch_norm_11, inplace=True)
        batch_norm_11 = None
        src_3 = torch.nn.functional.interpolate(
            xout_11, size=(160, 160), mode="bilinear"
        )
        xout_11 = None
        hx_9 = torch.cat((src_3, xout_2), 1)
        src_3 = xout_2 = None
        conv2d_13 = torch.conv2d(
            hx_9,
            l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_9 = (
            l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_12 = torch.nn.functional.batch_norm(
            conv2d_13,
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_13 = (
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_12 = torch.nn.functional.relu(batch_norm_12, inplace=True)
        batch_norm_12 = None
        src_4 = torch.nn.functional.interpolate(
            xout_12, size=(320, 320), mode="bilinear"
        )
        xout_12 = None
        hx_10 = torch.cat((src_4, xout_1), 1)
        src_4 = xout_1 = None
        conv2d_14 = torch.conv2d(
            hx_10,
            l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_10 = (
            l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_13 = torch.nn.functional.batch_norm(
            conv2d_14,
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_14 = (
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_13 = torch.nn.functional.relu(batch_norm_13, inplace=True)
        batch_norm_13 = None
        hx1 = xout_13 + xout
        xout_13 = xout = None
        hx_11 = torch.nn.functional.max_pool2d(
            hx1, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_15 = torch.conv2d(
            hx_11,
            l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_11 = (
            l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_14 = torch.nn.functional.batch_norm(
            conv2d_15,
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_15 = (
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_14 = torch.nn.functional.relu(batch_norm_14, inplace=True)
        batch_norm_14 = None
        conv2d_16 = torch.conv2d(
            xout_14,
            l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage2_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_15 = torch.nn.functional.batch_norm(
            conv2d_16,
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_16 = (
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_15 = torch.nn.functional.relu(batch_norm_15, inplace=True)
        batch_norm_15 = None
        hx_12 = torch.nn.functional.max_pool2d(
            xout_15, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_17 = torch.conv2d(
            hx_12,
            l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_12 = (
            l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_16 = torch.nn.functional.batch_norm(
            conv2d_17,
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_17 = (
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_16 = torch.nn.functional.relu(batch_norm_16, inplace=True)
        batch_norm_16 = None
        hx_13 = torch.nn.functional.max_pool2d(
            xout_16, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_18 = torch.conv2d(
            hx_13,
            l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_13 = (
            l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_17 = torch.nn.functional.batch_norm(
            conv2d_18,
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_18 = (
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_17 = torch.nn.functional.relu(batch_norm_17, inplace=True)
        batch_norm_17 = None
        hx_14 = torch.nn.functional.max_pool2d(
            xout_17, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_19 = torch.conv2d(
            hx_14,
            l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_14 = (
            l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_18 = torch.nn.functional.batch_norm(
            conv2d_19,
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_19 = (
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_18 = torch.nn.functional.relu(batch_norm_18, inplace=True)
        batch_norm_18 = None
        hx_15 = torch.nn.functional.max_pool2d(
            xout_18, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_20 = torch.conv2d(
            hx_15,
            l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_15 = (
            l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_19 = torch.nn.functional.batch_norm(
            conv2d_20,
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_20 = (
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_19 = torch.nn.functional.relu(batch_norm_19, inplace=True)
        batch_norm_19 = None
        conv2d_21 = torch.conv2d(
            xout_19,
            l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage2_modules_rebnconv6_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_20 = torch.nn.functional.batch_norm(
            conv2d_21,
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_21 = (
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv6_modules_bn_s1_parameters_bias_
        ) = None
        xout_20 = torch.nn.functional.relu(batch_norm_20, inplace=True)
        batch_norm_20 = None
        hx_16 = torch.cat((xout_20, xout_19), 1)
        xout_20 = xout_19 = None
        conv2d_22 = torch.conv2d(
            hx_16,
            l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_16 = (
            l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_21 = torch.nn.functional.batch_norm(
            conv2d_22,
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_22 = (
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        ) = None
        xout_21 = torch.nn.functional.relu(batch_norm_21, inplace=True)
        batch_norm_21 = None
        src_5 = torch.nn.functional.interpolate(xout_21, size=(20, 20), mode="bilinear")
        xout_21 = None
        hx_17 = torch.cat((src_5, xout_18), 1)
        src_5 = xout_18 = None
        conv2d_23 = torch.conv2d(
            hx_17,
            l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_17 = (
            l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_22 = torch.nn.functional.batch_norm(
            conv2d_23,
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_23 = (
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_22 = torch.nn.functional.relu(batch_norm_22, inplace=True)
        batch_norm_22 = None
        src_6 = torch.nn.functional.interpolate(xout_22, size=(40, 40), mode="bilinear")
        xout_22 = None
        hx_18 = torch.cat((src_6, xout_17), 1)
        src_6 = xout_17 = None
        conv2d_24 = torch.conv2d(
            hx_18,
            l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_18 = (
            l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_23 = torch.nn.functional.batch_norm(
            conv2d_24,
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_24 = (
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_23 = torch.nn.functional.relu(batch_norm_23, inplace=True)
        batch_norm_23 = None
        src_7 = torch.nn.functional.interpolate(xout_23, size=(80, 80), mode="bilinear")
        xout_23 = None
        hx_19 = torch.cat((src_7, xout_16), 1)
        src_7 = xout_16 = None
        conv2d_25 = torch.conv2d(
            hx_19,
            l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_19 = (
            l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_24 = torch.nn.functional.batch_norm(
            conv2d_25,
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_25 = (
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_24 = torch.nn.functional.relu(batch_norm_24, inplace=True)
        batch_norm_24 = None
        src_8 = torch.nn.functional.interpolate(
            xout_24, size=(160, 160), mode="bilinear"
        )
        xout_24 = None
        hx_20 = torch.cat((src_8, xout_15), 1)
        src_8 = xout_15 = None
        conv2d_26 = torch.conv2d(
            hx_20,
            l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_20 = (
            l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_25 = torch.nn.functional.batch_norm(
            conv2d_26,
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_26 = (
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_25 = torch.nn.functional.relu(batch_norm_25, inplace=True)
        batch_norm_25 = None
        hx2 = xout_25 + xout_14
        xout_25 = xout_14 = None
        hx_21 = torch.nn.functional.max_pool2d(
            hx2, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_27 = torch.conv2d(
            hx_21,
            l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_21 = (
            l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_26 = torch.nn.functional.batch_norm(
            conv2d_27,
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_27 = (
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_26 = torch.nn.functional.relu(batch_norm_26, inplace=True)
        batch_norm_26 = None
        conv2d_28 = torch.conv2d(
            xout_26,
            l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage3_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_27 = torch.nn.functional.batch_norm(
            conv2d_28,
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_28 = (
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_27 = torch.nn.functional.relu(batch_norm_27, inplace=True)
        batch_norm_27 = None
        hx_22 = torch.nn.functional.max_pool2d(
            xout_27, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_29 = torch.conv2d(
            hx_22,
            l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_22 = (
            l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_28 = torch.nn.functional.batch_norm(
            conv2d_29,
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_29 = (
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_28 = torch.nn.functional.relu(batch_norm_28, inplace=True)
        batch_norm_28 = None
        hx_23 = torch.nn.functional.max_pool2d(
            xout_28, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_30 = torch.conv2d(
            hx_23,
            l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_23 = (
            l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_29 = torch.nn.functional.batch_norm(
            conv2d_30,
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_30 = (
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_29 = torch.nn.functional.relu(batch_norm_29, inplace=True)
        batch_norm_29 = None
        hx_24 = torch.nn.functional.max_pool2d(
            xout_29, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_31 = torch.conv2d(
            hx_24,
            l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_24 = (
            l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_30 = torch.nn.functional.batch_norm(
            conv2d_31,
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_31 = (
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_30 = torch.nn.functional.relu(batch_norm_30, inplace=True)
        batch_norm_30 = None
        conv2d_32 = torch.conv2d(
            xout_30,
            l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage3_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_31 = torch.nn.functional.batch_norm(
            conv2d_32,
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_32 = (
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_31 = torch.nn.functional.relu(batch_norm_31, inplace=True)
        batch_norm_31 = None
        hx_25 = torch.cat((xout_31, xout_30), 1)
        xout_31 = xout_30 = None
        conv2d_33 = torch.conv2d(
            hx_25,
            l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_25 = (
            l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_32 = torch.nn.functional.batch_norm(
            conv2d_33,
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_33 = (
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_32 = torch.nn.functional.relu(batch_norm_32, inplace=True)
        batch_norm_32 = None
        src_9 = torch.nn.functional.interpolate(xout_32, size=(20, 20), mode="bilinear")
        xout_32 = None
        hx_26 = torch.cat((src_9, xout_29), 1)
        src_9 = xout_29 = None
        conv2d_34 = torch.conv2d(
            hx_26,
            l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_26 = (
            l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_33 = torch.nn.functional.batch_norm(
            conv2d_34,
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_34 = (
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_33 = torch.nn.functional.relu(batch_norm_33, inplace=True)
        batch_norm_33 = None
        src_10 = torch.nn.functional.interpolate(
            xout_33, size=(40, 40), mode="bilinear"
        )
        xout_33 = None
        hx_27 = torch.cat((src_10, xout_28), 1)
        src_10 = xout_28 = None
        conv2d_35 = torch.conv2d(
            hx_27,
            l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_27 = (
            l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_34 = torch.nn.functional.batch_norm(
            conv2d_35,
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_35 = (
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_34 = torch.nn.functional.relu(batch_norm_34, inplace=True)
        batch_norm_34 = None
        src_11 = torch.nn.functional.interpolate(
            xout_34, size=(80, 80), mode="bilinear"
        )
        xout_34 = None
        hx_28 = torch.cat((src_11, xout_27), 1)
        src_11 = xout_27 = None
        conv2d_36 = torch.conv2d(
            hx_28,
            l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_28 = (
            l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_35 = torch.nn.functional.batch_norm(
            conv2d_36,
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_36 = (
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_35 = torch.nn.functional.relu(batch_norm_35, inplace=True)
        batch_norm_35 = None
        hx3 = xout_35 + xout_26
        xout_35 = xout_26 = None
        hx_29 = torch.nn.functional.max_pool2d(
            hx3, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_37 = torch.conv2d(
            hx_29,
            l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_29 = (
            l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_36 = torch.nn.functional.batch_norm(
            conv2d_37,
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_37 = (
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_36 = torch.nn.functional.relu(batch_norm_36, inplace=True)
        batch_norm_36 = None
        conv2d_38 = torch.conv2d(
            xout_36,
            l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage4_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_37 = torch.nn.functional.batch_norm(
            conv2d_38,
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_38 = (
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_37 = torch.nn.functional.relu(batch_norm_37, inplace=True)
        batch_norm_37 = None
        hx_30 = torch.nn.functional.max_pool2d(
            xout_37, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_39 = torch.conv2d(
            hx_30,
            l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_30 = (
            l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_38 = torch.nn.functional.batch_norm(
            conv2d_39,
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_39 = (
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_38 = torch.nn.functional.relu(batch_norm_38, inplace=True)
        batch_norm_38 = None
        hx_31 = torch.nn.functional.max_pool2d(
            xout_38, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_40 = torch.conv2d(
            hx_31,
            l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_31 = (
            l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_39 = torch.nn.functional.batch_norm(
            conv2d_40,
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_40 = (
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_39 = torch.nn.functional.relu(batch_norm_39, inplace=True)
        batch_norm_39 = None
        conv2d_41 = torch.conv2d(
            xout_39,
            l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage4_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_40 = torch.nn.functional.batch_norm(
            conv2d_41,
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_41 = (
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_40 = torch.nn.functional.relu(batch_norm_40, inplace=True)
        batch_norm_40 = None
        hx_32 = torch.cat((xout_40, xout_39), 1)
        xout_40 = xout_39 = None
        conv2d_42 = torch.conv2d(
            hx_32,
            l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_32 = (
            l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_41 = torch.nn.functional.batch_norm(
            conv2d_42,
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_42 = (
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_41 = torch.nn.functional.relu(batch_norm_41, inplace=True)
        batch_norm_41 = None
        src_12 = torch.nn.functional.interpolate(
            xout_41, size=(20, 20), mode="bilinear"
        )
        xout_41 = None
        hx_33 = torch.cat((src_12, xout_38), 1)
        src_12 = xout_38 = None
        conv2d_43 = torch.conv2d(
            hx_33,
            l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_33 = (
            l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_42 = torch.nn.functional.batch_norm(
            conv2d_43,
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_43 = (
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_42 = torch.nn.functional.relu(batch_norm_42, inplace=True)
        batch_norm_42 = None
        src_13 = torch.nn.functional.interpolate(
            xout_42, size=(40, 40), mode="bilinear"
        )
        xout_42 = None
        hx_34 = torch.cat((src_13, xout_37), 1)
        src_13 = xout_37 = None
        conv2d_44 = torch.conv2d(
            hx_34,
            l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_34 = (
            l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_43 = torch.nn.functional.batch_norm(
            conv2d_44,
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_44 = (
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_43 = torch.nn.functional.relu(batch_norm_43, inplace=True)
        batch_norm_43 = None
        hx4 = xout_43 + xout_36
        xout_43 = xout_36 = None
        hx_35 = torch.nn.functional.max_pool2d(
            hx4, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_45 = torch.conv2d(
            hx_35,
            l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_35 = (
            l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_44 = torch.nn.functional.batch_norm(
            conv2d_45,
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_45 = (
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_44 = torch.nn.functional.relu(batch_norm_44, inplace=True)
        batch_norm_44 = None
        conv2d_46 = torch.conv2d(
            xout_44,
            l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_45 = torch.nn.functional.batch_norm(
            conv2d_46,
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_46 = (
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_45 = torch.nn.functional.relu(batch_norm_45, inplace=True)
        batch_norm_45 = None
        conv2d_47 = torch.conv2d(
            xout_45,
            l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_46 = torch.nn.functional.batch_norm(
            conv2d_47,
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_47 = (
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_46 = torch.nn.functional.relu(batch_norm_46, inplace=True)
        batch_norm_46 = None
        conv2d_48 = torch.conv2d(
            xout_46,
            l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_47 = torch.nn.functional.batch_norm(
            conv2d_48,
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_48 = (
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_47 = torch.nn.functional.relu(batch_norm_47, inplace=True)
        batch_norm_47 = None
        conv2d_49 = torch.conv2d(
            xout_47,
            l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (8, 8),
            (8, 8),
            1,
        )
        l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_48 = torch.nn.functional.batch_norm(
            conv2d_49,
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_49 = (
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_48 = torch.nn.functional.relu(batch_norm_48, inplace=True)
        batch_norm_48 = None
        hx_36 = torch.cat((xout_48, xout_47), 1)
        xout_48 = xout_47 = None
        conv2d_50 = torch.conv2d(
            hx_36,
            l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        hx_36 = (
            l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_49 = torch.nn.functional.batch_norm(
            conv2d_50,
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_50 = (
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_49 = torch.nn.functional.relu(batch_norm_49, inplace=True)
        batch_norm_49 = None
        hx_37 = torch.cat((xout_49, xout_46), 1)
        xout_49 = xout_46 = None
        conv2d_51 = torch.conv2d(
            hx_37,
            l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        hx_37 = (
            l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_50 = torch.nn.functional.batch_norm(
            conv2d_51,
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_51 = (
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_50 = torch.nn.functional.relu(batch_norm_50, inplace=True)
        batch_norm_50 = None
        hx_38 = torch.cat((xout_50, xout_45), 1)
        xout_50 = xout_45 = None
        conv2d_52 = torch.conv2d(
            hx_38,
            l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_38 = (
            l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_51 = torch.nn.functional.batch_norm(
            conv2d_52,
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_52 = (
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_51 = torch.nn.functional.relu(batch_norm_51, inplace=True)
        batch_norm_51 = None
        hx5 = xout_51 + xout_44
        xout_51 = xout_44 = None
        hx_39 = torch.nn.functional.max_pool2d(
            hx5, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_53 = torch.conv2d(
            hx_39,
            l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_39 = (
            l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_52 = torch.nn.functional.batch_norm(
            conv2d_53,
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_53 = (
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_52 = torch.nn.functional.relu(batch_norm_52, inplace=True)
        batch_norm_52 = None
        conv2d_54 = torch.conv2d(
            xout_52,
            l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage6_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_53 = torch.nn.functional.batch_norm(
            conv2d_54,
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_54 = (
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_53 = torch.nn.functional.relu(batch_norm_53, inplace=True)
        batch_norm_53 = None
        conv2d_55 = torch.conv2d(
            xout_53,
            l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage6_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_54 = torch.nn.functional.batch_norm(
            conv2d_55,
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_55 = (
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_54 = torch.nn.functional.relu(batch_norm_54, inplace=True)
        batch_norm_54 = None
        conv2d_56 = torch.conv2d(
            xout_54,
            l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage6_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_55 = torch.nn.functional.batch_norm(
            conv2d_56,
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_56 = (
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_55 = torch.nn.functional.relu(batch_norm_55, inplace=True)
        batch_norm_55 = None
        conv2d_57 = torch.conv2d(
            xout_55,
            l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (8, 8),
            (8, 8),
            1,
        )
        l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage6_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_56 = torch.nn.functional.batch_norm(
            conv2d_57,
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_57 = (
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_56 = torch.nn.functional.relu(batch_norm_56, inplace=True)
        batch_norm_56 = None
        hx_40 = torch.cat((xout_56, xout_55), 1)
        xout_56 = xout_55 = None
        conv2d_58 = torch.conv2d(
            hx_40,
            l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        hx_40 = (
            l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_57 = torch.nn.functional.batch_norm(
            conv2d_58,
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_58 = (
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_57 = torch.nn.functional.relu(batch_norm_57, inplace=True)
        batch_norm_57 = None
        hx_41 = torch.cat((xout_57, xout_54), 1)
        xout_57 = xout_54 = None
        conv2d_59 = torch.conv2d(
            hx_41,
            l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        hx_41 = (
            l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_58 = torch.nn.functional.batch_norm(
            conv2d_59,
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_59 = (
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_58 = torch.nn.functional.relu(batch_norm_58, inplace=True)
        batch_norm_58 = None
        hx_42 = torch.cat((xout_58, xout_53), 1)
        xout_58 = xout_53 = None
        conv2d_60 = torch.conv2d(
            hx_42,
            l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_42 = (
            l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_59 = torch.nn.functional.batch_norm(
            conv2d_60,
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_60 = (
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage6_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_59 = torch.nn.functional.relu(batch_norm_59, inplace=True)
        batch_norm_59 = None
        hx6 = xout_59 + xout_52
        xout_59 = xout_52 = None
        src_14 = torch.nn.functional.interpolate(hx6, size=(20, 20), mode="bilinear")
        hx_43 = torch.cat((src_14, hx5), 1)
        src_14 = hx5 = None
        conv2d_61 = torch.conv2d(
            hx_43,
            l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_43 = (
            l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_60 = torch.nn.functional.batch_norm(
            conv2d_61,
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_61 = l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_60 = torch.nn.functional.relu(batch_norm_60, inplace=True)
        batch_norm_60 = None
        conv2d_62 = torch.conv2d(
            xout_60,
            l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_61 = torch.nn.functional.batch_norm(
            conv2d_62,
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_62 = (
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_61 = torch.nn.functional.relu(batch_norm_61, inplace=True)
        batch_norm_61 = None
        conv2d_63 = torch.conv2d(
            xout_61,
            l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_62 = torch.nn.functional.batch_norm(
            conv2d_63,
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_63 = (
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_62 = torch.nn.functional.relu(batch_norm_62, inplace=True)
        batch_norm_62 = None
        conv2d_64 = torch.conv2d(
            xout_62,
            l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_63 = torch.nn.functional.batch_norm(
            conv2d_64,
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_64 = (
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_63 = torch.nn.functional.relu(batch_norm_63, inplace=True)
        batch_norm_63 = None
        conv2d_65 = torch.conv2d(
            xout_63,
            l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (8, 8),
            (8, 8),
            1,
        )
        l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage5d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_64 = torch.nn.functional.batch_norm(
            conv2d_65,
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_65 = (
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_64 = torch.nn.functional.relu(batch_norm_64, inplace=True)
        batch_norm_64 = None
        hx_44 = torch.cat((xout_64, xout_63), 1)
        xout_64 = xout_63 = None
        conv2d_66 = torch.conv2d(
            hx_44,
            l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (4, 4),
            (4, 4),
            1,
        )
        hx_44 = (
            l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_65 = torch.nn.functional.batch_norm(
            conv2d_66,
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_66 = l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_65 = torch.nn.functional.relu(batch_norm_65, inplace=True)
        batch_norm_65 = None
        hx_45 = torch.cat((xout_65, xout_62), 1)
        xout_65 = xout_62 = None
        conv2d_67 = torch.conv2d(
            hx_45,
            l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        hx_45 = (
            l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_66 = torch.nn.functional.batch_norm(
            conv2d_67,
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_67 = l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_66 = torch.nn.functional.relu(batch_norm_66, inplace=True)
        batch_norm_66 = None
        hx_46 = torch.cat((xout_66, xout_61), 1)
        xout_66 = xout_61 = None
        conv2d_68 = torch.conv2d(
            hx_46,
            l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_46 = (
            l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_67 = torch.nn.functional.batch_norm(
            conv2d_68,
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_68 = l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage5d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_67 = torch.nn.functional.relu(batch_norm_67, inplace=True)
        batch_norm_67 = None
        hx5d = xout_67 + xout_60
        xout_67 = xout_60 = None
        src_15 = torch.nn.functional.interpolate(hx5d, size=(40, 40), mode="bilinear")
        hx_47 = torch.cat((src_15, hx4), 1)
        src_15 = hx4 = None
        conv2d_69 = torch.conv2d(
            hx_47,
            l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_47 = (
            l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_68 = torch.nn.functional.batch_norm(
            conv2d_69,
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_69 = l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_68 = torch.nn.functional.relu(batch_norm_68, inplace=True)
        batch_norm_68 = None
        conv2d_70 = torch.conv2d(
            xout_68,
            l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage4d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_69 = torch.nn.functional.batch_norm(
            conv2d_70,
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_70 = (
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_69 = torch.nn.functional.relu(batch_norm_69, inplace=True)
        batch_norm_69 = None
        hx_48 = torch.nn.functional.max_pool2d(
            xout_69, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_71 = torch.conv2d(
            hx_48,
            l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_48 = (
            l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_70 = torch.nn.functional.batch_norm(
            conv2d_71,
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_71 = (
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_70 = torch.nn.functional.relu(batch_norm_70, inplace=True)
        batch_norm_70 = None
        hx_49 = torch.nn.functional.max_pool2d(
            xout_70, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_72 = torch.conv2d(
            hx_49,
            l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_49 = (
            l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_71 = torch.nn.functional.batch_norm(
            conv2d_72,
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_72 = (
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_71 = torch.nn.functional.relu(batch_norm_71, inplace=True)
        batch_norm_71 = None
        conv2d_73 = torch.conv2d(
            xout_71,
            l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage4d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_72 = torch.nn.functional.batch_norm(
            conv2d_73,
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_73 = (
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_72 = torch.nn.functional.relu(batch_norm_72, inplace=True)
        batch_norm_72 = None
        hx_50 = torch.cat((xout_72, xout_71), 1)
        xout_72 = xout_71 = None
        conv2d_74 = torch.conv2d(
            hx_50,
            l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_50 = (
            l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_73 = torch.nn.functional.batch_norm(
            conv2d_74,
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_74 = l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_73 = torch.nn.functional.relu(batch_norm_73, inplace=True)
        batch_norm_73 = None
        src_16 = torch.nn.functional.interpolate(
            xout_73, size=(20, 20), mode="bilinear"
        )
        xout_73 = None
        hx_51 = torch.cat((src_16, xout_70), 1)
        src_16 = xout_70 = None
        conv2d_75 = torch.conv2d(
            hx_51,
            l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_51 = (
            l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_74 = torch.nn.functional.batch_norm(
            conv2d_75,
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_75 = l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_74 = torch.nn.functional.relu(batch_norm_74, inplace=True)
        batch_norm_74 = None
        src_17 = torch.nn.functional.interpolate(
            xout_74, size=(40, 40), mode="bilinear"
        )
        xout_74 = None
        hx_52 = torch.cat((src_17, xout_69), 1)
        src_17 = xout_69 = None
        conv2d_76 = torch.conv2d(
            hx_52,
            l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_52 = (
            l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_75 = torch.nn.functional.batch_norm(
            conv2d_76,
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_76 = l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage4d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_75 = torch.nn.functional.relu(batch_norm_75, inplace=True)
        batch_norm_75 = None
        hx4d = xout_75 + xout_68
        xout_75 = xout_68 = None
        src_18 = torch.nn.functional.interpolate(hx4d, size=(80, 80), mode="bilinear")
        hx_53 = torch.cat((src_18, hx3), 1)
        src_18 = hx3 = None
        conv2d_77 = torch.conv2d(
            hx_53,
            l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_53 = (
            l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_76 = torch.nn.functional.batch_norm(
            conv2d_77,
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_77 = l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_76 = torch.nn.functional.relu(batch_norm_76, inplace=True)
        batch_norm_76 = None
        conv2d_78 = torch.conv2d(
            xout_76,
            l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage3d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_77 = torch.nn.functional.batch_norm(
            conv2d_78,
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_78 = (
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_77 = torch.nn.functional.relu(batch_norm_77, inplace=True)
        batch_norm_77 = None
        hx_54 = torch.nn.functional.max_pool2d(
            xout_77, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_79 = torch.conv2d(
            hx_54,
            l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_54 = (
            l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_78 = torch.nn.functional.batch_norm(
            conv2d_79,
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_79 = (
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_78 = torch.nn.functional.relu(batch_norm_78, inplace=True)
        batch_norm_78 = None
        hx_55 = torch.nn.functional.max_pool2d(
            xout_78, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_80 = torch.conv2d(
            hx_55,
            l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_55 = (
            l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_79 = torch.nn.functional.batch_norm(
            conv2d_80,
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_80 = (
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_79 = torch.nn.functional.relu(batch_norm_79, inplace=True)
        batch_norm_79 = None
        hx_56 = torch.nn.functional.max_pool2d(
            xout_79, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_81 = torch.conv2d(
            hx_56,
            l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_56 = (
            l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_80 = torch.nn.functional.batch_norm(
            conv2d_81,
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_81 = (
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_80 = torch.nn.functional.relu(batch_norm_80, inplace=True)
        batch_norm_80 = None
        conv2d_82 = torch.conv2d(
            xout_80,
            l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage3d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_81 = torch.nn.functional.batch_norm(
            conv2d_82,
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_82 = (
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_81 = torch.nn.functional.relu(batch_norm_81, inplace=True)
        batch_norm_81 = None
        hx_57 = torch.cat((xout_81, xout_80), 1)
        xout_81 = xout_80 = None
        conv2d_83 = torch.conv2d(
            hx_57,
            l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_57 = (
            l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_82 = torch.nn.functional.batch_norm(
            conv2d_83,
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_83 = l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_82 = torch.nn.functional.relu(batch_norm_82, inplace=True)
        batch_norm_82 = None
        src_19 = torch.nn.functional.interpolate(
            xout_82, size=(20, 20), mode="bilinear"
        )
        xout_82 = None
        hx_58 = torch.cat((src_19, xout_79), 1)
        src_19 = xout_79 = None
        conv2d_84 = torch.conv2d(
            hx_58,
            l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_58 = (
            l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_83 = torch.nn.functional.batch_norm(
            conv2d_84,
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_84 = l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_83 = torch.nn.functional.relu(batch_norm_83, inplace=True)
        batch_norm_83 = None
        src_20 = torch.nn.functional.interpolate(
            xout_83, size=(40, 40), mode="bilinear"
        )
        xout_83 = None
        hx_59 = torch.cat((src_20, xout_78), 1)
        src_20 = xout_78 = None
        conv2d_85 = torch.conv2d(
            hx_59,
            l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_59 = (
            l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_84 = torch.nn.functional.batch_norm(
            conv2d_85,
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_85 = l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_84 = torch.nn.functional.relu(batch_norm_84, inplace=True)
        batch_norm_84 = None
        src_21 = torch.nn.functional.interpolate(
            xout_84, size=(80, 80), mode="bilinear"
        )
        xout_84 = None
        hx_60 = torch.cat((src_21, xout_77), 1)
        src_21 = xout_77 = None
        conv2d_86 = torch.conv2d(
            hx_60,
            l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_60 = (
            l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_85 = torch.nn.functional.batch_norm(
            conv2d_86,
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_86 = l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage3d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_85 = torch.nn.functional.relu(batch_norm_85, inplace=True)
        batch_norm_85 = None
        hx3d = xout_85 + xout_76
        xout_85 = xout_76 = None
        src_22 = torch.nn.functional.interpolate(hx3d, size=(160, 160), mode="bilinear")
        hx_61 = torch.cat((src_22, hx2), 1)
        src_22 = hx2 = None
        conv2d_87 = torch.conv2d(
            hx_61,
            l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_61 = (
            l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_86 = torch.nn.functional.batch_norm(
            conv2d_87,
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_87 = l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_86 = torch.nn.functional.relu(batch_norm_86, inplace=True)
        batch_norm_86 = None
        conv2d_88 = torch.conv2d(
            xout_86,
            l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage2d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_87 = torch.nn.functional.batch_norm(
            conv2d_88,
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_88 = (
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_87 = torch.nn.functional.relu(batch_norm_87, inplace=True)
        batch_norm_87 = None
        hx_62 = torch.nn.functional.max_pool2d(
            xout_87, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_89 = torch.conv2d(
            hx_62,
            l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_62 = (
            l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_88 = torch.nn.functional.batch_norm(
            conv2d_89,
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_89 = (
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_88 = torch.nn.functional.relu(batch_norm_88, inplace=True)
        batch_norm_88 = None
        hx_63 = torch.nn.functional.max_pool2d(
            xout_88, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_90 = torch.conv2d(
            hx_63,
            l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_63 = (
            l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_89 = torch.nn.functional.batch_norm(
            conv2d_90,
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_90 = (
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_89 = torch.nn.functional.relu(batch_norm_89, inplace=True)
        batch_norm_89 = None
        hx_64 = torch.nn.functional.max_pool2d(
            xout_89, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_91 = torch.conv2d(
            hx_64,
            l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_64 = (
            l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_90 = torch.nn.functional.batch_norm(
            conv2d_91,
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_91 = (
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_90 = torch.nn.functional.relu(batch_norm_90, inplace=True)
        batch_norm_90 = None
        hx_65 = torch.nn.functional.max_pool2d(
            xout_90, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_92 = torch.conv2d(
            hx_65,
            l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_65 = (
            l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_91 = torch.nn.functional.batch_norm(
            conv2d_92,
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_92 = (
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_91 = torch.nn.functional.relu(batch_norm_91, inplace=True)
        batch_norm_91 = None
        conv2d_93 = torch.conv2d(
            xout_91,
            l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage2d_modules_rebnconv6_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_92 = torch.nn.functional.batch_norm(
            conv2d_93,
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_93 = (
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv6_modules_bn_s1_parameters_bias_
        ) = None
        xout_92 = torch.nn.functional.relu(batch_norm_92, inplace=True)
        batch_norm_92 = None
        hx_66 = torch.cat((xout_92, xout_91), 1)
        xout_92 = xout_91 = None
        conv2d_94 = torch.conv2d(
            hx_66,
            l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_66 = (
            l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_93 = torch.nn.functional.batch_norm(
            conv2d_94,
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_94 = l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        ) = None
        xout_93 = torch.nn.functional.relu(batch_norm_93, inplace=True)
        batch_norm_93 = None
        src_23 = torch.nn.functional.interpolate(
            xout_93, size=(20, 20), mode="bilinear"
        )
        xout_93 = None
        hx_67 = torch.cat((src_23, xout_90), 1)
        src_23 = xout_90 = None
        conv2d_95 = torch.conv2d(
            hx_67,
            l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_67 = (
            l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_94 = torch.nn.functional.batch_norm(
            conv2d_95,
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_95 = l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_94 = torch.nn.functional.relu(batch_norm_94, inplace=True)
        batch_norm_94 = None
        src_24 = torch.nn.functional.interpolate(
            xout_94, size=(40, 40), mode="bilinear"
        )
        xout_94 = None
        hx_68 = torch.cat((src_24, xout_89), 1)
        src_24 = xout_89 = None
        conv2d_96 = torch.conv2d(
            hx_68,
            l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_68 = (
            l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_95 = torch.nn.functional.batch_norm(
            conv2d_96,
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_96 = l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_95 = torch.nn.functional.relu(batch_norm_95, inplace=True)
        batch_norm_95 = None
        src_25 = torch.nn.functional.interpolate(
            xout_95, size=(80, 80), mode="bilinear"
        )
        xout_95 = None
        hx_69 = torch.cat((src_25, xout_88), 1)
        src_25 = xout_88 = None
        conv2d_97 = torch.conv2d(
            hx_69,
            l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_69 = (
            l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_96 = torch.nn.functional.batch_norm(
            conv2d_97,
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_97 = l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_96 = torch.nn.functional.relu(batch_norm_96, inplace=True)
        batch_norm_96 = None
        src_26 = torch.nn.functional.interpolate(
            xout_96, size=(160, 160), mode="bilinear"
        )
        xout_96 = None
        hx_70 = torch.cat((src_26, xout_87), 1)
        src_26 = xout_87 = None
        conv2d_98 = torch.conv2d(
            hx_70,
            l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_70 = (
            l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_97 = torch.nn.functional.batch_norm(
            conv2d_98,
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_98 = l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage2d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_97 = torch.nn.functional.relu(batch_norm_97, inplace=True)
        batch_norm_97 = None
        hx2d = xout_97 + xout_86
        xout_97 = xout_86 = None
        src_27 = torch.nn.functional.interpolate(hx2d, size=(320, 320), mode="bilinear")
        hx_71 = torch.cat((src_27, hx1), 1)
        src_27 = hx1 = None
        conv2d_99 = torch.conv2d(
            hx_71,
            l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_71 = (
            l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconvin_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_98 = torch.nn.functional.batch_norm(
            conv2d_99,
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_99 = l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconvin_modules_bn_s1_parameters_bias_
        ) = None
        xout_98 = torch.nn.functional.relu(batch_norm_98, inplace=True)
        batch_norm_98 = None
        conv2d_100 = torch.conv2d(
            xout_98,
            l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage1d_modules_rebnconv1_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_99 = torch.nn.functional.batch_norm(
            conv2d_100,
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_100 = (
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1_modules_bn_s1_parameters_bias_
        ) = None
        xout_99 = torch.nn.functional.relu(batch_norm_99, inplace=True)
        batch_norm_99 = None
        hx_72 = torch.nn.functional.max_pool2d(
            xout_99, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_101 = torch.conv2d(
            hx_72,
            l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_72 = (
            l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_100 = torch.nn.functional.batch_norm(
            conv2d_101,
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_101 = (
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2_modules_bn_s1_parameters_bias_
        ) = None
        xout_100 = torch.nn.functional.relu(batch_norm_100, inplace=True)
        batch_norm_100 = None
        hx_73 = torch.nn.functional.max_pool2d(
            xout_100, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_102 = torch.conv2d(
            hx_73,
            l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_73 = (
            l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_101 = torch.nn.functional.batch_norm(
            conv2d_102,
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_102 = (
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3_modules_bn_s1_parameters_bias_
        ) = None
        xout_101 = torch.nn.functional.relu(batch_norm_101, inplace=True)
        batch_norm_101 = None
        hx_74 = torch.nn.functional.max_pool2d(
            xout_101, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_103 = torch.conv2d(
            hx_74,
            l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_74 = (
            l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_102 = torch.nn.functional.batch_norm(
            conv2d_103,
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_103 = (
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4_modules_bn_s1_parameters_bias_
        ) = None
        xout_102 = torch.nn.functional.relu(batch_norm_102, inplace=True)
        batch_norm_102 = None
        hx_75 = torch.nn.functional.max_pool2d(
            xout_102, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_104 = torch.conv2d(
            hx_75,
            l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_75 = (
            l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_103 = torch.nn.functional.batch_norm(
            conv2d_104,
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_104 = (
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5_modules_bn_s1_parameters_bias_
        ) = None
        xout_103 = torch.nn.functional.relu(batch_norm_103, inplace=True)
        batch_norm_103 = None
        hx_76 = torch.nn.functional.max_pool2d(
            xout_103, 2, 2, 0, 1, ceil_mode=True, return_indices=False
        )
        conv2d_105 = torch.conv2d(
            hx_76,
            l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_76 = (
            l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_104 = torch.nn.functional.batch_norm(
            conv2d_105,
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_105 = (
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6_modules_bn_s1_parameters_bias_
        ) = None
        xout_104 = torch.nn.functional.relu(batch_norm_104, inplace=True)
        batch_norm_104 = None
        conv2d_106 = torch.conv2d(
            xout_104,
            l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_weight_ = (
            l_self_modules_stage1d_modules_rebnconv7_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_105 = torch.nn.functional.batch_norm(
            conv2d_106,
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_106 = (
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_mean_
        ) = (
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv7_modules_bn_s1_parameters_bias_
        ) = None
        xout_105 = torch.nn.functional.relu(batch_norm_105, inplace=True)
        batch_norm_105 = None
        hx_77 = torch.cat((xout_105, xout_104), 1)
        xout_105 = xout_104 = None
        conv2d_107 = torch.conv2d(
            hx_77,
            l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_77 = (
            l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_106 = torch.nn.functional.batch_norm(
            conv2d_107,
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_107 = l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv6d_modules_bn_s1_parameters_bias_
        ) = None
        xout_106 = torch.nn.functional.relu(batch_norm_106, inplace=True)
        batch_norm_106 = None
        src_28 = torch.nn.functional.interpolate(
            xout_106, size=(20, 20), mode="bilinear"
        )
        xout_106 = None
        hx_78 = torch.cat((src_28, xout_103), 1)
        src_28 = xout_103 = None
        conv2d_108 = torch.conv2d(
            hx_78,
            l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_78 = (
            l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_107 = torch.nn.functional.batch_norm(
            conv2d_108,
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_108 = l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv5d_modules_bn_s1_parameters_bias_
        ) = None
        xout_107 = torch.nn.functional.relu(batch_norm_107, inplace=True)
        batch_norm_107 = None
        src_29 = torch.nn.functional.interpolate(
            xout_107, size=(40, 40), mode="bilinear"
        )
        xout_107 = None
        hx_79 = torch.cat((src_29, xout_102), 1)
        src_29 = xout_102 = None
        conv2d_109 = torch.conv2d(
            hx_79,
            l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_79 = (
            l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_108 = torch.nn.functional.batch_norm(
            conv2d_109,
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_109 = l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv4d_modules_bn_s1_parameters_bias_
        ) = None
        xout_108 = torch.nn.functional.relu(batch_norm_108, inplace=True)
        batch_norm_108 = None
        src_30 = torch.nn.functional.interpolate(
            xout_108, size=(80, 80), mode="bilinear"
        )
        xout_108 = None
        hx_80 = torch.cat((src_30, xout_101), 1)
        src_30 = xout_101 = None
        conv2d_110 = torch.conv2d(
            hx_80,
            l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_80 = (
            l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_109 = torch.nn.functional.batch_norm(
            conv2d_110,
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_110 = l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv3d_modules_bn_s1_parameters_bias_
        ) = None
        xout_109 = torch.nn.functional.relu(batch_norm_109, inplace=True)
        batch_norm_109 = None
        src_31 = torch.nn.functional.interpolate(
            xout_109, size=(160, 160), mode="bilinear"
        )
        xout_109 = None
        hx_81 = torch.cat((src_31, xout_100), 1)
        src_31 = xout_100 = None
        conv2d_111 = torch.conv2d(
            hx_81,
            l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_81 = (
            l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_110 = torch.nn.functional.batch_norm(
            conv2d_111,
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_111 = l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv2d_modules_bn_s1_parameters_bias_
        ) = None
        xout_110 = torch.nn.functional.relu(batch_norm_110, inplace=True)
        batch_norm_110 = None
        src_32 = torch.nn.functional.interpolate(
            xout_110, size=(320, 320), mode="bilinear"
        )
        xout_110 = None
        hx_82 = torch.cat((src_32, xout_99), 1)
        src_32 = xout_99 = None
        conv2d_112 = torch.conv2d(
            hx_82,
            l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        hx_82 = (
            l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1d_modules_conv_s1_parameters_bias_
        ) = None
        batch_norm_111 = torch.nn.functional.batch_norm(
            conv2d_112,
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_,
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_,
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_,
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        conv2d_112 = l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_mean_ = (
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_buffers_running_var_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_weight_
        ) = (
            l_self_modules_stage1d_modules_rebnconv1d_modules_bn_s1_parameters_bias_
        ) = None
        xout_111 = torch.nn.functional.relu(batch_norm_111, inplace=True)
        batch_norm_111 = None
        hx1d = xout_111 + xout_98
        xout_111 = xout_98 = None
        d1 = torch.conv2d(
            hx1d,
            l_self_modules_side1_parameters_weight_,
            l_self_modules_side1_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side1_parameters_weight_ = (
            l_self_modules_side1_parameters_bias_
        ) = None
        src_33 = torch.nn.functional.interpolate(d1, size=(640, 640), mode="bilinear")
        d1 = None
        d2 = torch.conv2d(
            hx2d,
            l_self_modules_side2_parameters_weight_,
            l_self_modules_side2_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side2_parameters_weight_ = (
            l_self_modules_side2_parameters_bias_
        ) = None
        src_34 = torch.nn.functional.interpolate(d2, size=(640, 640), mode="bilinear")
        d2 = None
        d3 = torch.conv2d(
            hx3d,
            l_self_modules_side3_parameters_weight_,
            l_self_modules_side3_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side3_parameters_weight_ = (
            l_self_modules_side3_parameters_bias_
        ) = None
        src_35 = torch.nn.functional.interpolate(d3, size=(640, 640), mode="bilinear")
        d3 = None
        d4 = torch.conv2d(
            hx4d,
            l_self_modules_side4_parameters_weight_,
            l_self_modules_side4_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side4_parameters_weight_ = (
            l_self_modules_side4_parameters_bias_
        ) = None
        src_36 = torch.nn.functional.interpolate(d4, size=(640, 640), mode="bilinear")
        d4 = None
        d5 = torch.conv2d(
            hx5d,
            l_self_modules_side5_parameters_weight_,
            l_self_modules_side5_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side5_parameters_weight_ = (
            l_self_modules_side5_parameters_bias_
        ) = None
        src_37 = torch.nn.functional.interpolate(d5, size=(640, 640), mode="bilinear")
        d5 = None
        d6 = torch.conv2d(
            hx6,
            l_self_modules_side6_parameters_weight_,
            l_self_modules_side6_parameters_bias_,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        l_self_modules_side6_parameters_weight_ = (
            l_self_modules_side6_parameters_bias_
        ) = None
        src_38 = torch.nn.functional.interpolate(d6, size=(640, 640), mode="bilinear")
        d6 = None
        sigmoid = torch.nn.functional.sigmoid(src_33)
        src_33 = None
        sigmoid_1 = torch.nn.functional.sigmoid(src_34)
        src_34 = None
        sigmoid_2 = torch.nn.functional.sigmoid(src_35)
        src_35 = None
        sigmoid_3 = torch.nn.functional.sigmoid(src_36)
        src_36 = None
        sigmoid_4 = torch.nn.functional.sigmoid(src_37)
        src_37 = None
        sigmoid_5 = torch.nn.functional.sigmoid(src_38)
        src_38 = None
        return (
            sigmoid,
            sigmoid_1,
            sigmoid_2,
            sigmoid_3,
            sigmoid_4,
            sigmoid_5,
            hx1d,
            hx2d,
            hx3d,
            hx4d,
            hx5d,
            hx6,
        )
