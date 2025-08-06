import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stem_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_stem_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_stem_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stem_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stem_modules_bn_buffers_running_mean_ = (
            L_self_modules_stem_modules_bn_buffers_running_mean_
        )
        l_self_modules_stem_modules_bn_buffers_running_var_ = (
            L_self_modules_stem_modules_bn_buffers_running_var_
        )
        l_self_modules_stem_modules_bn_parameters_weight_ = (
            L_self_modules_stem_modules_bn_parameters_weight_
        )
        l_self_modules_stem_modules_bn_parameters_bias_ = (
            L_self_modules_stem_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_
        l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = L_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_
        l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_ = L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_
        l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_stem_modules_bn_buffers_running_mean_,
            l_self_modules_stem_modules_bn_buffers_running_var_,
            l_self_modules_stem_modules_bn_parameters_weight_,
            l_self_modules_stem_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_stem_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_stem_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_stem_modules_bn_parameters_weight_
        ) = l_self_modules_stem_modules_bn_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s1_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_5 = torch.nn.functional.relu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            3,
        )
        x_5 = (
            l_self_modules_s1_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_7 = torch.nn.functional.batch_norm(
            x_6,
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_6 = (
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_8 = torch.nn.functional.relu(x_7, inplace=True)
        x_7 = None
        x_9 = torch.conv2d(
            x_8,
            l_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_8 = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_10 = torch.nn.functional.batch_norm(
            x_9,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_9 = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_11 = torch.conv2d(
            x_2,
            l_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_13 = x_10 + x_12
        x_10 = x_12 = None
        x_14 = torch.nn.functional.relu(x_13, inplace=False)
        x_13 = None
        x_15 = torch.conv2d(
            x_14,
            l_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s1_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_17 = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_23 = x_22 + x_14
        x_22 = x_14 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=False)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            7,
        )
        x_27 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_30,
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_30 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_33 = torch.conv2d(
            x_24,
            l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_24 = l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_35 = x_32 + x_34
        x_32 = x_34 = None
        x_36 = torch.nn.functional.relu(x_35, inplace=False)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
            l_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        x_39 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_45 = x_44 + x_36
        x_44 = x_36 = None
        x_46 = torch.nn.functional.relu(x_45, inplace=False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        x_49 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_55 = x_54 + x_46
        x_54 = x_46 = None
        x_56 = torch.nn.functional.relu(x_55, inplace=False)
        x_55 = None
        x_57 = torch.conv2d(
            x_56,
            l_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_58 = torch.nn.functional.batch_norm(
            x_57,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_57 = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_59 = torch.nn.functional.relu(x_58, inplace=True)
        x_58 = None
        x_60 = torch.conv2d(
            x_59,
            l_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            7,
        )
        x_59 = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_61 = torch.nn.functional.batch_norm(
            x_60,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_60 = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_62 = torch.nn.functional.relu(x_61, inplace=True)
        x_61 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_65 = x_64 + x_56
        x_64 = x_56 = None
        x_66 = torch.nn.functional.relu(x_65, inplace=False)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            17,
        )
        x_69 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_72 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_75 = torch.conv2d(
            x_66,
            l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_77 = x_74 + x_76
        x_74 = x_76 = None
        x_78 = torch.nn.functional.relu(x_77, inplace=False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
            l_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_81 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_87 = x_86 + x_78
        x_86 = x_78 = None
        x_88 = torch.nn.functional.relu(x_87, inplace=False)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_91 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_97 = x_96 + x_88
        x_96 = x_88 = None
        x_98 = torch.nn.functional.relu(x_97, inplace=False)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
            l_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_101 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_107 = x_106 + x_98
        x_106 = x_98 = None
        x_108 = torch.nn.functional.relu(x_107, inplace=False)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_111 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_117 = x_116 + x_108
        x_116 = x_108 = None
        x_118 = torch.nn.functional.relu(x_117, inplace=False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_121 = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_126 = torch.nn.functional.batch_norm(
            x_125,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_125 = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_127 = x_126 + x_118
        x_126 = x_118 = None
        x_128 = torch.nn.functional.relu(x_127, inplace=False)
        x_127 = None
        x_129 = torch.conv2d(
            x_128,
            l_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b7_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_130 = torch.nn.functional.batch_norm(
            x_129,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_129 = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_131 = torch.nn.functional.relu(x_130, inplace=True)
        x_130 = None
        x_132 = torch.conv2d(
            x_131,
            l_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_131 = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_134 = torch.nn.functional.relu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_134 = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_137 = x_136 + x_128
        x_136 = x_128 = None
        x_138 = torch.nn.functional.relu(x_137, inplace=False)
        x_137 = None
        x_139 = torch.conv2d(
            x_138,
            l_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b8_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_141 = torch.nn.functional.relu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_141 = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_143 = torch.nn.functional.batch_norm(
            x_142,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_142 = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_144 = torch.nn.functional.relu(x_143, inplace=True)
        x_143 = None
        x_145 = torch.conv2d(
            x_144,
            l_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_144 = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_146 = torch.nn.functional.batch_norm(
            x_145,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_145 = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_147 = x_146 + x_138
        x_146 = x_138 = None
        x_148 = torch.nn.functional.relu(x_147, inplace=False)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b9_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_150 = torch.nn.functional.batch_norm(
            x_149,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_149 = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_151 = torch.nn.functional.relu(x_150, inplace=True)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
            l_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_151 = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_157 = x_156 + x_148
        x_156 = x_148 = None
        x_158 = torch.nn.functional.relu(x_157, inplace=False)
        x_157 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b10_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_161 = torch.nn.functional.relu(x_160, inplace=True)
        x_160 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_161 = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_164 = torch.nn.functional.relu(x_163, inplace=True)
        x_163 = None
        x_165 = torch.conv2d(
            x_164,
            l_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_164 = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_167 = x_166 + x_158
        x_166 = x_158 = None
        x_168 = torch.nn.functional.relu(x_167, inplace=False)
        x_167 = None
        x_169 = torch.conv2d(
            x_168,
            l_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            38,
        )
        x_171 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_177 = torch.conv2d(
            x_168,
            l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_168 = l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_179 = x_176 + x_178
        x_176 = x_178 = None
        x_180 = torch.nn.functional.relu(x_179, inplace=False)
        x_179 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        x_184 = torch.conv2d(
            x_183,
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            38,
        )
        x_183 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            x_186,
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_186 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_189 = x_188 + x_180
        x_188 = x_180 = None
        x_190 = torch.nn.functional.relu(x_189, inplace=False)
        x_189 = None
        x_191 = torch.nn.functional.adaptive_avg_pool2d(x_190, 1)
        x_190 = None
        x_192 = x_191.flatten(1, -1)
        x_191 = None
        x_193 = torch.nn.functional.dropout(x_192, 0.0, False, False)
        x_192 = None
        x_194 = torch._C._nn.linear(
            x_193,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_193 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_194,)
