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
        L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_
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
        l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_
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
        l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_
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
        x_se = x_8.mean((2, 3), keepdim=True)
        x_se_1 = torch.conv2d(
            x_se,
            l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se = (
            l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_2 = torch.nn.functional.relu(x_se_1, inplace=True)
        x_se_1 = None
        x_se_3 = torch.conv2d(
            x_se_2,
            l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_2 = (
            l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid = x_se_3.sigmoid()
        x_se_3 = None
        x_9 = x_8 * sigmoid
        x_8 = sigmoid = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_12 = torch.conv2d(
            x_2,
            l_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_s1_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_13 = torch.nn.functional.batch_norm(
            x_12,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_12 = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s1_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_14 = x_11 + x_13
        x_11 = x_13 = None
        x_15 = torch.nn.functional.relu(x_14, inplace=False)
        x_14 = None
        x_16 = torch.conv2d(
            x_15,
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
        x_17 = torch.nn.functional.batch_norm(
            x_16,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_16 = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_18 = torch.nn.functional.relu(x_17, inplace=True)
        x_17 = None
        x_19 = torch.conv2d(
            x_18,
            l_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        x_18 = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_20 = torch.nn.functional.batch_norm(
            x_19,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_19 = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        x_se_4 = x_21.mean((2, 3), keepdim=True)
        x_se_5 = torch.conv2d(
            x_se_4,
            l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_4 = (
            l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_6 = torch.nn.functional.relu(x_se_5, inplace=True)
        x_se_5 = None
        x_se_7 = torch.conv2d(
            x_se_6,
            l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_6 = (
            l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_1 = x_se_7.sigmoid()
        x_se_7 = None
        x_22 = x_21 * sigmoid_1
        x_21 = sigmoid_1 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_22 = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s1_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_25 = x_24 + x_15
        x_24 = x_15 = None
        x_26 = torch.nn.functional.relu(x_25, inplace=False)
        x_25 = None
        x_27 = torch.conv2d(
            x_26,
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
        x_28 = torch.nn.functional.batch_norm(
            x_27,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_27 = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_29 = torch.nn.functional.relu(x_28, inplace=True)
        x_28 = None
        x_30 = torch.conv2d(
            x_29,
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            3,
        )
        x_29 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_31 = torch.nn.functional.batch_norm(
            x_30,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_30 = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_32 = torch.nn.functional.relu(x_31, inplace=True)
        x_31 = None
        x_se_8 = x_32.mean((2, 3), keepdim=True)
        x_se_9 = torch.conv2d(
            x_se_8,
            l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_8 = (
            l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_10 = torch.nn.functional.relu(x_se_9, inplace=True)
        x_se_9 = None
        x_se_11 = torch.conv2d(
            x_se_10,
            l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_10 = (
            l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_2 = x_se_11.sigmoid()
        x_se_11 = None
        x_33 = x_32 * sigmoid_2
        x_32 = sigmoid_2 = None
        x_34 = torch.conv2d(
            x_33,
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_33 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_35 = torch.nn.functional.batch_norm(
            x_34,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_34 = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_36 = torch.conv2d(
            x_26,
            l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_26 = l_self_modules_s2_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s2_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_38 = x_35 + x_37
        x_35 = x_37 = None
        x_39 = torch.nn.functional.relu(x_38, inplace=False)
        x_38 = None
        x_40 = torch.conv2d(
            x_39,
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
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_42 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_se_12 = x_45.mean((2, 3), keepdim=True)
        x_se_13 = torch.conv2d(
            x_se_12,
            l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_12 = (
            l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_14 = torch.nn.functional.relu(x_se_13, inplace=True)
        x_se_13 = None
        x_se_15 = torch.conv2d(
            x_se_14,
            l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_14 = (
            l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_3 = x_se_15.sigmoid()
        x_se_15 = None
        x_46 = x_45 * sigmoid_3
        x_45 = sigmoid_3 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_49 = x_48 + x_39
        x_48 = x_39 = None
        x_50 = torch.nn.functional.relu(x_49, inplace=False)
        x_49 = None
        x_51 = torch.conv2d(
            x_50,
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
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_53 = torch.nn.functional.relu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_53 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_56 = torch.nn.functional.relu(x_55, inplace=True)
        x_55 = None
        x_se_16 = x_56.mean((2, 3), keepdim=True)
        x_se_17 = torch.conv2d(
            x_se_16,
            l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_16 = (
            l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_ = None
        x_se_18 = torch.nn.functional.relu(x_se_17, inplace=True)
        x_se_17 = None
        x_se_19 = torch.conv2d(
            x_se_18,
            l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_18 = (
            l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_4 = x_se_19.sigmoid()
        x_se_19 = None
        x_57 = x_56 * sigmoid_4
        x_56 = sigmoid_4 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_60 = x_59 + x_50
        x_59 = x_50 = None
        x_61 = torch.nn.functional.relu(x_60, inplace=False)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
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
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_64 = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_se_20 = x_67.mean((2, 3), keepdim=True)
        x_se_21 = torch.conv2d(
            x_se_20,
            l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_20 = (
            l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_ = None
        x_se_22 = torch.nn.functional.relu(x_se_21, inplace=True)
        x_se_21 = None
        x_se_23 = torch.conv2d(
            x_se_22,
            l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_22 = (
            l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_5 = x_se_23.sigmoid()
        x_se_23 = None
        x_68 = x_67 * sigmoid_5
        x_67 = sigmoid_5 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_68 = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_71 = x_70 + x_61
        x_70 = x_61 = None
        x_72 = torch.nn.functional.relu(x_71, inplace=False)
        x_71 = None
        x_73 = torch.conv2d(
            x_72,
            l_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_75 = (
            l_self_modules_s2_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = (
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_se_24 = x_78.mean((2, 3), keepdim=True)
        x_se_25 = torch.conv2d(
            x_se_24,
            l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_24 = (
            l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_ = None
        x_se_26 = torch.nn.functional.relu(x_se_25, inplace=True)
        x_se_25 = None
        x_se_27 = torch.conv2d(
            x_se_26,
            l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_26 = (
            l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_6 = x_se_27.sigmoid()
        x_se_27 = None
        x_79 = x_78 * sigmoid_6
        x_78 = sigmoid_6 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = (
            l_self_modules_s2_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_82 = x_81 + x_72
        x_81 = x_72 = None
        x_83 = torch.nn.functional.relu(x_82, inplace=False)
        x_82 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b6_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_86 = torch.nn.functional.relu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            3,
        )
        x_86 = (
            l_self_modules_s2_modules_b6_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_88 = torch.nn.functional.batch_norm(
            x_87,
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_87 = (
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_89 = torch.nn.functional.relu(x_88, inplace=True)
        x_88 = None
        x_se_28 = x_89.mean((2, 3), keepdim=True)
        x_se_29 = torch.conv2d(
            x_se_28,
            l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_28 = (
            l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_ = None
        x_se_30 = torch.nn.functional.relu(x_se_29, inplace=True)
        x_se_29 = None
        x_se_31 = torch.conv2d(
            x_se_30,
            l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_30 = (
            l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_7 = x_se_31.sigmoid()
        x_se_31 = None
        x_90 = x_89 * sigmoid_7
        x_89 = sigmoid_7 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_90 = (
            l_self_modules_s2_modules_b6_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s2_modules_b6_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_93 = x_92 + x_83
        x_92 = x_83 = None
        x_94 = torch.nn.functional.relu(x_93, inplace=False)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
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
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        x_97 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_se_32 = x_100.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = (
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_34 = torch.nn.functional.relu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = (
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_101 = x_100 * sigmoid_8
        x_100 = sigmoid_8 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_101 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_104 = torch.conv2d(
            x_94,
            l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_s3_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s3_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_106 = x_103 + x_105
        x_103 = x_105 = None
        x_107 = torch.nn.functional.relu(x_106, inplace=False)
        x_106 = None
        x_108 = torch.conv2d(
            x_107,
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
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_110 = torch.nn.functional.relu(x_109, inplace=True)
        x_109 = None
        x_111 = torch.conv2d(
            x_110,
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_110 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_112 = torch.nn.functional.batch_norm(
            x_111,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_111 = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_113 = torch.nn.functional.relu(x_112, inplace=True)
        x_112 = None
        x_se_36 = x_113.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = (
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_38 = torch.nn.functional.relu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = (
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_114 = x_113 * sigmoid_9
        x_113 = sigmoid_9 = None
        x_115 = torch.conv2d(
            x_114,
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_114 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_117 = x_116 + x_107
        x_116 = x_107 = None
        x_118 = torch.nn.functional.relu(x_117, inplace=False)
        x_117 = None
        x_119 = torch.conv2d(
            x_118,
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
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_121 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_se_40 = x_124.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = (
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_ = None
        x_se_42 = torch.nn.functional.relu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = (
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_125 = x_124 * sigmoid_10
        x_124 = sigmoid_10 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_128 = x_127 + x_118
        x_127 = x_118 = None
        x_129 = torch.nn.functional.relu(x_128, inplace=False)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
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
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_132 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        x_se_44 = x_135.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = (
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_ = None
        x_se_46 = torch.nn.functional.relu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = (
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_136 = x_135 * sigmoid_11
        x_135 = sigmoid_11 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_139 = x_138 + x_129
        x_138 = x_129 = None
        x_140 = torch.nn.functional.relu(x_139, inplace=False)
        x_139 = None
        x_141 = torch.conv2d(
            x_140,
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
        x_142 = torch.nn.functional.batch_norm(
            x_141,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_141 = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_143 = torch.nn.functional.relu(x_142, inplace=True)
        x_142 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_143 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_146 = torch.nn.functional.relu(x_145, inplace=True)
        x_145 = None
        x_se_48 = x_146.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = (
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_ = None
        x_se_50 = torch.nn.functional.relu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = (
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_147 = x_146 * sigmoid_12
        x_146 = sigmoid_12 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_150 = x_149 + x_140
        x_149 = x_140 = None
        x_151 = torch.nn.functional.relu(x_150, inplace=False)
        x_150 = None
        x_152 = torch.conv2d(
            x_151,
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
        x_153 = torch.nn.functional.batch_norm(
            x_152,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_152 = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_154 = torch.nn.functional.relu(x_153, inplace=True)
        x_153 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_154 = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_se_52 = x_157.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = (
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_ = None
        x_se_54 = torch.nn.functional.relu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = (
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_158 = x_157 * sigmoid_13
        x_157 = sigmoid_13 = None
        x_159 = torch.conv2d(
            x_158,
            l_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_158 = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_160 = torch.nn.functional.batch_norm(
            x_159,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_159 = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b6_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_161 = x_160 + x_151
        x_160 = x_151 = None
        x_162 = torch.nn.functional.relu(x_161, inplace=False)
        x_161 = None
        x_163 = torch.conv2d(
            x_162,
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
        x_164 = torch.nn.functional.batch_norm(
            x_163,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_163 = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        x_166 = torch.conv2d(
            x_165,
            l_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_165 = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_se_56 = x_168.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = (
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_ = None
        x_se_58 = torch.nn.functional.relu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = (
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_169 = x_168 * sigmoid_14
        x_168 = sigmoid_14 = None
        x_170 = torch.conv2d(
            x_169,
            l_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_169 = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_171 = torch.nn.functional.batch_norm(
            x_170,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_170 = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b7_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_172 = x_171 + x_162
        x_171 = x_162 = None
        x_173 = torch.nn.functional.relu(x_172, inplace=False)
        x_172 = None
        x_174 = torch.conv2d(
            x_173,
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
        x_175 = torch.nn.functional.batch_norm(
            x_174,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_174 = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_176 = torch.nn.functional.relu(x_175, inplace=True)
        x_175 = None
        x_177 = torch.conv2d(
            x_176,
            l_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_176 = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_178 = torch.nn.functional.batch_norm(
            x_177,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_177 = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_179 = torch.nn.functional.relu(x_178, inplace=True)
        x_178 = None
        x_se_60 = x_179.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = (
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_ = None
        x_se_62 = torch.nn.functional.relu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = (
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_180 = x_179 * sigmoid_15
        x_179 = sigmoid_15 = None
        x_181 = torch.conv2d(
            x_180,
            l_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_180 = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_182 = torch.nn.functional.batch_norm(
            x_181,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_181 = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b8_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_183 = x_182 + x_173
        x_182 = x_173 = None
        x_184 = torch.nn.functional.relu(x_183, inplace=False)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
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
        x_186 = torch.nn.functional.batch_norm(
            x_185,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_185 = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_187 = torch.nn.functional.relu(x_186, inplace=True)
        x_186 = None
        x_188 = torch.conv2d(
            x_187,
            l_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_187 = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_189 = torch.nn.functional.batch_norm(
            x_188,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_188 = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_190 = torch.nn.functional.relu(x_189, inplace=True)
        x_189 = None
        x_se_64 = x_190.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = (
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_ = None
        x_se_66 = torch.nn.functional.relu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = (
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_191 = x_190 * sigmoid_16
        x_190 = sigmoid_16 = None
        x_192 = torch.conv2d(
            x_191,
            l_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_191 = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_193 = torch.nn.functional.batch_norm(
            x_192,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_192 = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b9_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_194 = x_193 + x_184
        x_193 = x_184 = None
        x_195 = torch.nn.functional.relu(x_194, inplace=False)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
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
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        x_199 = torch.conv2d(
            x_198,
            l_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_198 = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        x_se_68 = x_201.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = (
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_ = None
        x_se_70 = torch.nn.functional.relu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = (
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_202 = x_201 * sigmoid_17
        x_201 = sigmoid_17 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b10_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_205 = x_204 + x_195
        x_204 = x_195 = None
        x_206 = torch.nn.functional.relu(x_205, inplace=False)
        x_205 = None
        x_207 = torch.conv2d(
            x_206,
            l_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b11_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_208 = torch.nn.functional.batch_norm(
            x_207,
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_207 = (
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_209 = torch.nn.functional.relu(x_208, inplace=True)
        x_208 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_209 = (
            l_self_modules_s3_modules_b11_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = (
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_212 = torch.nn.functional.relu(x_211, inplace=True)
        x_211 = None
        x_se_72 = x_212.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = (
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_ = None
        x_se_74 = torch.nn.functional.relu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = (
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_213 = x_212 * sigmoid_18
        x_212 = sigmoid_18 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_213 = (
            l_self_modules_s3_modules_b11_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = (
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b11_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_216 = x_215 + x_206
        x_215 = x_206 = None
        x_217 = torch.nn.functional.relu(x_216, inplace=False)
        x_216 = None
        x_218 = torch.conv2d(
            x_217,
            l_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b12_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_219 = torch.nn.functional.batch_norm(
            x_218,
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_218 = (
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_220 = torch.nn.functional.relu(x_219, inplace=True)
        x_219 = None
        x_221 = torch.conv2d(
            x_220,
            l_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_220 = (
            l_self_modules_s3_modules_b12_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_222 = torch.nn.functional.batch_norm(
            x_221,
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_221 = (
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_223 = torch.nn.functional.relu(x_222, inplace=True)
        x_222 = None
        x_se_76 = x_223.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = (
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_ = None
        x_se_78 = torch.nn.functional.relu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = (
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_224 = x_223 * sigmoid_19
        x_223 = sigmoid_19 = None
        x_225 = torch.conv2d(
            x_224,
            l_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_224 = (
            l_self_modules_s3_modules_b12_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_226 = torch.nn.functional.batch_norm(
            x_225,
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_225 = (
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s3_modules_b12_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_227 = x_226 + x_217
        x_226 = x_217 = None
        x_228 = torch.nn.functional.relu(x_227, inplace=False)
        x_227 = None
        x_229 = torch.conv2d(
            x_228,
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
        x_230 = torch.nn.functional.batch_norm(
            x_229,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_229 = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_231 = torch.nn.functional.relu(x_230, inplace=True)
        x_230 = None
        x_232 = torch.conv2d(
            x_231,
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            17,
        )
        x_231 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_233 = torch.nn.functional.batch_norm(
            x_232,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_232 = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        x_se_80 = x_234.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = (
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_82 = torch.nn.functional.relu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = (
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_235 = x_234 * sigmoid_20
        x_234 = sigmoid_20 = None
        x_236 = torch.conv2d(
            x_235,
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_235 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_237 = torch.nn.functional.batch_norm(
            x_236,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_236 = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_238 = torch.conv2d(
            x_228,
            l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_228 = l_self_modules_s4_modules_b1_modules_downsample_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_mean_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_buffers_running_var_ = l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_weight_ = (
            l_self_modules_s4_modules_b1_modules_downsample_modules_bn_parameters_bias_
        ) = None
        x_240 = x_237 + x_239
        x_237 = x_239 = None
        x_241 = torch.nn.functional.relu(x_240, inplace=False)
        x_240 = None
        x_242 = torch.conv2d(
            x_241,
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
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_244 = torch.nn.functional.relu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            17,
        )
        x_244 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_conv_parameters_weight_
        ) = None
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_se_84 = x_247.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = (
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_86 = torch.nn.functional.relu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = (
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_248 = x_247 * sigmoid_21
        x_247 = sigmoid_21 = None
        x_249 = torch.conv2d(
            x_248,
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_248 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_conv_parameters_weight_
        ) = None
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_s4_modules_b2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_251 = x_250 + x_241
        x_250 = x_241 = None
        x_252 = torch.nn.functional.relu(x_251, inplace=False)
        x_251 = None
        x_253 = torch.nn.functional.adaptive_avg_pool2d(x_252, 1)
        x_252 = None
        x_254 = x_253.flatten(1, -1)
        x_253 = None
        x_255 = torch.nn.functional.dropout(x_254, 0.0, False, False)
        x_254 = None
        x_256 = torch._C._nn.linear(
            x_255,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_255 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_256,)
