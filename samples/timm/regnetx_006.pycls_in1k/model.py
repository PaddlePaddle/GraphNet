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
            2,
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
            4,
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
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_29 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_33 = torch.conv2d(
            x_32,
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_32 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_35 = x_34 + x_26
        x_34 = x_26 = None
        x_36 = torch.nn.functional.relu(x_35, inplace=False)
        x_35 = None
        x_37 = torch.conv2d(
            x_36,
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
        x_38 = torch.nn.functional.batch_norm(
            x_37,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_37 = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_39 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_45 = x_44 + x_36
        x_44 = x_36 = None
        x_46 = torch.nn.functional.relu(x_45, inplace=False)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
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
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            10,
        )
        x_49 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_54 = torch.nn.functional.batch_norm(
            x_53,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_53 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_55 = torch.conv2d(
            x_46,
            l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_57 = x_54 + x_56
        x_54 = x_56 = None
        x_58 = torch.nn.functional.relu(x_57, inplace=False)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
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
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        x_61 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_67 = x_66 + x_58
        x_66 = x_58 = None
        x_68 = torch.nn.functional.relu(x_67, inplace=False)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
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
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_71 = torch.nn.functional.relu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        x_71 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_74 = torch.nn.functional.relu(x_73, inplace=True)
        x_73 = None
        x_75 = torch.conv2d(
            x_74,
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_74 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_76 = torch.nn.functional.batch_norm(
            x_75,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_75 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_77 = x_76 + x_68
        x_76 = x_68 = None
        x_78 = torch.nn.functional.relu(x_77, inplace=False)
        x_77 = None
        x_79 = torch.conv2d(
            x_78,
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
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        x_82 = torch.conv2d(
            x_81,
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        x_81 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_83 = torch.nn.functional.batch_norm(
            x_82,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_82 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_84 = torch.nn.functional.relu(x_83, inplace=True)
        x_83 = None
        x_85 = torch.conv2d(
            x_84,
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_84 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_86 = torch.nn.functional.batch_norm(
            x_85,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_85 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_87 = x_86 + x_78
        x_86 = x_78 = None
        x_88 = torch.nn.functional.relu(x_87, inplace=False)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
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
        x_90 = torch.nn.functional.batch_norm(
            x_89,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_89 = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            10,
        )
        x_91 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_97 = x_96 + x_88
        x_96 = x_88 = None
        x_98 = torch.nn.functional.relu(x_97, inplace=False)
        x_97 = None
        x_99 = torch.conv2d(
            x_98,
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
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_101 = torch.nn.functional.relu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            22,
        )
        x_101 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_104 = torch.nn.functional.relu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_107 = torch.conv2d(
            x_98,
            l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_98 = l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_108 = torch.nn.functional.batch_norm(
            x_107,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_107 = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_109 = x_106 + x_108
        x_106 = x_108 = None
        x_110 = torch.nn.functional.relu(x_109, inplace=False)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
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
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_114 = torch.conv2d(
            x_113,
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_113 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_115 = torch.nn.functional.batch_norm(
            x_114,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_114 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_116 = torch.nn.functional.relu(x_115, inplace=True)
        x_115 = None
        x_117 = torch.conv2d(
            x_116,
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_116 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_119 = x_118 + x_110
        x_118 = x_110 = None
        x_120 = torch.nn.functional.relu(x_119, inplace=False)
        x_119 = None
        x_121 = torch.conv2d(
            x_120,
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
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_123 = torch.nn.functional.relu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_123 = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_126 = torch.nn.functional.relu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_126 = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_128 = torch.nn.functional.batch_norm(
            x_127,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_127 = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_129 = x_128 + x_120
        x_128 = x_120 = None
        x_130 = torch.nn.functional.relu(x_129, inplace=False)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
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
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_133 = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_135 = torch.nn.functional.batch_norm(
            x_134,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_134 = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_139 = x_138 + x_130
        x_138 = x_130 = None
        x_140 = torch.nn.functional.relu(x_139, inplace=False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
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
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_143 = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_149 = x_148 + x_140
        x_148 = x_140 = None
        x_150 = torch.nn.functional.relu(x_149, inplace=False)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
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
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_153 = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_156 = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b6_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_159 = x_158 + x_150
        x_158 = x_150 = None
        x_160 = torch.nn.functional.relu(x_159, inplace=False)
        x_159 = None
        x_161 = torch.conv2d(
            x_160,
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
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            22,
        )
        x_163 = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_165 = torch.nn.functional.batch_norm(
            x_164,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_164 = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_166 = torch.nn.functional.relu(x_165, inplace=True)
        x_165 = None
        x_167 = torch.conv2d(
            x_166,
            l_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_166 = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_168 = torch.nn.functional.batch_norm(
            x_167,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_167 = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b7_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_169 = x_168 + x_160
        x_168 = x_160 = None
        x_170 = torch.nn.functional.relu(x_169, inplace=False)
        x_169 = None
        x_171 = torch.nn.functional.adaptive_avg_pool2d(x_170, 1)
        x_170 = None
        x_172 = x_171.flatten(1, -1)
        x_171 = None
        x_173 = torch.nn.functional.dropout(x_172, 0.0, False, False)
        x_172 = None
        x_174 = torch._C._nn.linear(
            x_173,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_173 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_174,)
