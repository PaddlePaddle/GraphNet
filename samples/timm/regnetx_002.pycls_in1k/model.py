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
        L_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_
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
        x_16 = torch.nn.functional.batch_norm(
            x_15,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_15 = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_17 = torch.nn.functional.relu(x_16, inplace=True)
        x_16 = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            7,
        )
        x_17 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_20 = torch.nn.functional.relu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_23 = torch.conv2d(
            x_14,
            l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_14 = l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_25 = x_22 + x_24
        x_22 = x_24 = None
        x_26 = torch.nn.functional.relu(x_25, inplace=False)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
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
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            19,
        )
        x_29 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_35 = torch.conv2d(
            x_26,
            l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_37 = x_34 + x_36
        x_34 = x_36 = None
        x_38 = torch.nn.functional.relu(x_37, inplace=False)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
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
        x_40 = torch.nn.functional.batch_norm(
            x_39,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_39 = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_41 = torch.nn.functional.relu(x_40, inplace=True)
        x_40 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            19,
        )
        x_41 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_44 = torch.nn.functional.relu(x_43, inplace=True)
        x_43 = None
        x_45 = torch.conv2d(
            x_44,
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_44 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_46 = torch.nn.functional.batch_norm(
            x_45,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_45 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_47 = x_46 + x_38
        x_46 = x_38 = None
        x_48 = torch.nn.functional.relu(x_47, inplace=False)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
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
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_51,
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            19,
        )
        x_51 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        x_55 = torch.conv2d(
            x_54,
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_54 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_57 = x_56 + x_48
        x_56 = x_48 = None
        x_58 = torch.nn.functional.relu(x_57, inplace=False)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
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
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            19,
        )
        x_61 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_67 = x_66 + x_58
        x_66 = x_58 = None
        x_68 = torch.nn.functional.relu(x_67, inplace=False)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
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
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            46,
        )
        x_71 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_77 = torch.conv2d(
            x_68,
            l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_79 = x_76 + x_78
        x_76 = x_78 = None
        x_80 = torch.nn.functional.relu(x_79, inplace=False)
        x_79 = None
        x_81 = torch.conv2d(
            x_80,
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
        x_82 = torch.nn.functional.batch_norm(
            x_81,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_81 = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_83 = torch.nn.functional.relu(x_82, inplace=True)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_83 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_89 = x_88 + x_80
        x_88 = x_80 = None
        x_90 = torch.nn.functional.relu(x_89, inplace=False)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_93 = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            x_96,
            l_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_96 = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_99 = x_98 + x_90
        x_98 = x_90 = None
        x_100 = torch.nn.functional.relu(x_99, inplace=False)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_103 = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_109 = x_108 + x_100
        x_108 = x_100 = None
        x_110 = torch.nn.functional.relu(x_109, inplace=False)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_113 = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_119 = x_118 + x_110
        x_118 = x_110 = None
        x_120 = torch.nn.functional.relu(x_119, inplace=False)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
            l_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_123 = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_129 = x_128 + x_120
        x_128 = x_120 = None
        x_130 = torch.nn.functional.relu(x_129, inplace=False)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b7_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            46,
        )
        x_133 = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_139 = x_138 + x_130
        x_138 = x_130 = None
        x_140 = torch.nn.functional.relu(x_139, inplace=False)
        x_139 = None
        x_141 = torch.nn.functional.adaptive_avg_pool2d(x_140, 1)
        x_140 = None
        x_142 = x_141.flatten(1, -1)
        x_141 = None
        x_143 = torch.nn.functional.dropout(x_142, 0.0, False, False)
        x_142 = None
        x_144 = torch._C._nn.linear(
            x_143,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_143 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_144,)
