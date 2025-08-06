import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s1_modules_b2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s1_modules_b2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b3_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b4_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b5_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b6_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s2_modules_b7_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s2_modules_b7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b3_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b4_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b5_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b6_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b7_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b8_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b9_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b10_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b11_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b12_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b13_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s3_modules_b14_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s3_modules_b14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b1_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_s4_modules_b2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_s4_modules_b2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_parameters_weight_ = L_self_modules_stem_parameters_weight_
        l_x_ = L_x_
        l_self_modules_s1_modules_b1_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_norm1_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_norm1_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_norm1_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_norm1_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_norm1_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_norm1_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv1_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv1_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_norm2_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_norm2_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_norm2_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_norm2_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_norm2_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_norm2_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv2_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv2_parameters_weight_
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
        l_self_modules_s1_modules_b1_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s1_modules_b1_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s1_modules_b1_modules_norm3_buffers_running_var_ = (
            L_self_modules_s1_modules_b1_modules_norm3_buffers_running_var_
        )
        l_self_modules_s1_modules_b1_modules_norm3_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_norm3_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_norm3_parameters_bias_ = (
            L_self_modules_s1_modules_b1_modules_norm3_parameters_bias_
        )
        l_self_modules_s1_modules_b1_modules_conv3_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_conv3_parameters_weight_
        )
        l_self_modules_s1_modules_b1_modules_downsample_modules_1_parameters_weight_ = (
            L_self_modules_s1_modules_b1_modules_downsample_modules_1_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_norm1_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_norm1_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_norm1_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_norm1_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_norm1_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_norm1_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv1_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv1_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_norm2_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_norm2_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_norm2_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_norm2_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_norm2_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_norm2_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv2_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv2_parameters_weight_
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
        l_self_modules_s1_modules_b2_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s1_modules_b2_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s1_modules_b2_modules_norm3_buffers_running_var_ = (
            L_self_modules_s1_modules_b2_modules_norm3_buffers_running_var_
        )
        l_self_modules_s1_modules_b2_modules_norm3_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_norm3_parameters_weight_
        )
        l_self_modules_s1_modules_b2_modules_norm3_parameters_bias_ = (
            L_self_modules_s1_modules_b2_modules_norm3_parameters_bias_
        )
        l_self_modules_s1_modules_b2_modules_conv3_parameters_weight_ = (
            L_self_modules_s1_modules_b2_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b1_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b1_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b1_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b1_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b1_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b1_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b1_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b1_modules_downsample_modules_1_parameters_weight_ = (
            L_self_modules_s2_modules_b1_modules_downsample_modules_1_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b2_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b2_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b2_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b2_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b2_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b2_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b2_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b2_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b2_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b3_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b3_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b3_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b3_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b3_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b3_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b3_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b3_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b3_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b4_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b4_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b4_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b4_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b4_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b4_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b4_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b4_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b4_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b5_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b5_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b5_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b5_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b5_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b5_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b5_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b5_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b5_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv2_parameters_weight_
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
        l_self_modules_s2_modules_b6_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b6_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b6_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b6_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b6_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b6_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b6_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b6_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b6_modules_conv3_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s2_modules_b7_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s2_modules_b7_modules_norm1_buffers_running_var_ = (
            L_self_modules_s2_modules_b7_modules_norm1_buffers_running_var_
        )
        l_self_modules_s2_modules_b7_modules_norm1_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_norm1_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_norm1_parameters_bias_ = (
            L_self_modules_s2_modules_b7_modules_norm1_parameters_bias_
        )
        l_self_modules_s2_modules_b7_modules_conv1_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_conv1_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s2_modules_b7_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s2_modules_b7_modules_norm2_buffers_running_var_ = (
            L_self_modules_s2_modules_b7_modules_norm2_buffers_running_var_
        )
        l_self_modules_s2_modules_b7_modules_norm2_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_norm2_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_norm2_parameters_bias_ = (
            L_self_modules_s2_modules_b7_modules_norm2_parameters_bias_
        )
        l_self_modules_s2_modules_b7_modules_conv2_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_conv2_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s2_modules_b7_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s2_modules_b7_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s2_modules_b7_modules_norm3_buffers_running_var_ = (
            L_self_modules_s2_modules_b7_modules_norm3_buffers_running_var_
        )
        l_self_modules_s2_modules_b7_modules_norm3_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_norm3_parameters_weight_
        )
        l_self_modules_s2_modules_b7_modules_norm3_parameters_bias_ = (
            L_self_modules_s2_modules_b7_modules_norm3_parameters_bias_
        )
        l_self_modules_s2_modules_b7_modules_conv3_parameters_weight_ = (
            L_self_modules_s2_modules_b7_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b1_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b1_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b1_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b1_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b1_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b1_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b1_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b1_modules_downsample_modules_1_parameters_weight_ = (
            L_self_modules_s3_modules_b1_modules_downsample_modules_1_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b2_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b2_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b2_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b2_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b2_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b2_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b2_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b2_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b2_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b3_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b3_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b3_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b3_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b3_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b3_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b3_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b3_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b3_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b4_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b4_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b4_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b4_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b4_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b4_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b4_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b4_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b4_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b5_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b5_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b5_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b5_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b5_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b5_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b5_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b5_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b5_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b6_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b6_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b6_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b6_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b6_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b6_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b6_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b6_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b6_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b7_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b7_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b7_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b7_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b7_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b7_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b7_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b7_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b7_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b8_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b8_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b8_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b8_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b8_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b8_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b8_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b8_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b8_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b9_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b9_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b9_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b9_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b9_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b9_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b9_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b9_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b9_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b10_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b10_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b10_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b10_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b10_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b10_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b10_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b10_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b10_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b11_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b11_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b11_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b11_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b11_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b11_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b11_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b11_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b11_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv2_parameters_weight_
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
        l_self_modules_s3_modules_b12_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b12_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b12_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b12_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b12_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b12_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b12_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b12_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b12_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b13_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b13_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b13_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b13_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b13_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b13_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b13_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b13_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b13_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b13_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b13_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b13_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_conv2_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s3_modules_b13_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b13_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b13_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b13_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b13_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b13_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b13_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b13_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b13_modules_conv3_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s3_modules_b14_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s3_modules_b14_modules_norm1_buffers_running_var_ = (
            L_self_modules_s3_modules_b14_modules_norm1_buffers_running_var_
        )
        l_self_modules_s3_modules_b14_modules_norm1_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_norm1_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_norm1_parameters_bias_ = (
            L_self_modules_s3_modules_b14_modules_norm1_parameters_bias_
        )
        l_self_modules_s3_modules_b14_modules_conv1_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_conv1_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s3_modules_b14_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s3_modules_b14_modules_norm2_buffers_running_var_ = (
            L_self_modules_s3_modules_b14_modules_norm2_buffers_running_var_
        )
        l_self_modules_s3_modules_b14_modules_norm2_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_norm2_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_norm2_parameters_bias_ = (
            L_self_modules_s3_modules_b14_modules_norm2_parameters_bias_
        )
        l_self_modules_s3_modules_b14_modules_conv2_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_conv2_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_ = (
            L_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_
        )
        l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_ = (
            L_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_
        )
        l_self_modules_s3_modules_b14_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s3_modules_b14_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s3_modules_b14_modules_norm3_buffers_running_var_ = (
            L_self_modules_s3_modules_b14_modules_norm3_buffers_running_var_
        )
        l_self_modules_s3_modules_b14_modules_norm3_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_norm3_parameters_weight_
        )
        l_self_modules_s3_modules_b14_modules_norm3_parameters_bias_ = (
            L_self_modules_s3_modules_b14_modules_norm3_parameters_bias_
        )
        l_self_modules_s3_modules_b14_modules_conv3_parameters_weight_ = (
            L_self_modules_s3_modules_b14_modules_conv3_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_norm1_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_norm1_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_norm1_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_norm1_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_norm1_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_norm1_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv1_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv1_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_norm2_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_norm2_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_norm2_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_norm2_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_norm2_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_norm2_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv2_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv2_parameters_weight_
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
        l_self_modules_s4_modules_b1_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s4_modules_b1_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s4_modules_b1_modules_norm3_buffers_running_var_ = (
            L_self_modules_s4_modules_b1_modules_norm3_buffers_running_var_
        )
        l_self_modules_s4_modules_b1_modules_norm3_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_norm3_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_norm3_parameters_bias_ = (
            L_self_modules_s4_modules_b1_modules_norm3_parameters_bias_
        )
        l_self_modules_s4_modules_b1_modules_conv3_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_conv3_parameters_weight_
        )
        l_self_modules_s4_modules_b1_modules_downsample_modules_1_parameters_weight_ = (
            L_self_modules_s4_modules_b1_modules_downsample_modules_1_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_norm1_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_norm1_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_norm1_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_norm1_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_norm1_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_norm1_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_norm1_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_norm1_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv1_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv1_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_norm2_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_norm2_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_norm2_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_norm2_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_norm2_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_norm2_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_norm2_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_norm2_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv2_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv2_parameters_weight_
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
        l_self_modules_s4_modules_b2_modules_norm3_buffers_running_mean_ = (
            L_self_modules_s4_modules_b2_modules_norm3_buffers_running_mean_
        )
        l_self_modules_s4_modules_b2_modules_norm3_buffers_running_var_ = (
            L_self_modules_s4_modules_b2_modules_norm3_buffers_running_var_
        )
        l_self_modules_s4_modules_b2_modules_norm3_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_norm3_parameters_weight_
        )
        l_self_modules_s4_modules_b2_modules_norm3_parameters_bias_ = (
            L_self_modules_s4_modules_b2_modules_norm3_parameters_bias_
        )
        l_self_modules_s4_modules_b2_modules_conv3_parameters_weight_ = (
            L_self_modules_s4_modules_b2_modules_conv3_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        x = torch.conv2d(
            l_x_,
            l_self_modules_stem_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_s1_modules_b1_modules_norm1_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_norm1_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_norm1_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_s1_modules_b1_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm1_parameters_weight_
        ) = l_self_modules_s1_modules_b1_modules_norm1_parameters_bias_ = None
        x_2 = torch.nn.functional.silu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.conv2d(
            x_2,
            l_self_modules_s1_modules_b1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s1_modules_b1_modules_conv1_parameters_weight_ = None
        x_4 = torch.nn.functional.batch_norm(
            x_3,
            l_self_modules_s1_modules_b1_modules_norm2_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_norm2_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_norm2_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_3 = (
            l_self_modules_s1_modules_b1_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm2_parameters_weight_
        ) = l_self_modules_s1_modules_b1_modules_norm2_parameters_bias_ = None
        x_5 = torch.nn.functional.silu(x_4, inplace=True)
        x_4 = None
        x_6 = torch.conv2d(
            x_5,
            l_self_modules_s1_modules_b1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            2,
        )
        x_5 = l_self_modules_s1_modules_b1_modules_conv2_parameters_weight_ = None
        x_se = x_6.mean((2, 3), keepdim=True)
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
        x_se_2 = torch.nn.functional.silu(x_se_1, inplace=True)
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
        x_7 = x_6 * sigmoid
        x_6 = sigmoid = None
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_s1_modules_b1_modules_norm3_buffers_running_mean_,
            l_self_modules_s1_modules_b1_modules_norm3_buffers_running_var_,
            l_self_modules_s1_modules_b1_modules_norm3_parameters_weight_,
            l_self_modules_s1_modules_b1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = (
            l_self_modules_s1_modules_b1_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b1_modules_norm3_parameters_weight_
        ) = l_self_modules_s1_modules_b1_modules_norm3_parameters_bias_ = None
        x_9 = torch.nn.functional.silu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_9,
            l_self_modules_s1_modules_b1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_9 = l_self_modules_s1_modules_b1_modules_conv3_parameters_weight_ = None
        input_1 = torch._C._nn.avg_pool2d(x_2, 2, 2, 0, True, False, None)
        x_2 = None
        input_2 = torch.conv2d(
            input_1,
            l_self_modules_s1_modules_b1_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_1 = (
            l_self_modules_s1_modules_b1_modules_downsample_modules_1_parameters_weight_
        ) = None
        x_11 = x_10 + input_2
        x_10 = input_2 = None
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_s1_modules_b2_modules_norm1_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_norm1_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_norm1_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = (
            l_self_modules_s1_modules_b2_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm1_parameters_weight_
        ) = l_self_modules_s1_modules_b2_modules_norm1_parameters_bias_ = None
        x_13 = torch.nn.functional.silu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_s1_modules_b2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s1_modules_b2_modules_conv1_parameters_weight_ = None
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_s1_modules_b2_modules_norm2_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_norm2_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_norm2_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = (
            l_self_modules_s1_modules_b2_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm2_parameters_weight_
        ) = l_self_modules_s1_modules_b2_modules_norm2_parameters_bias_ = None
        x_16 = torch.nn.functional.silu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_s1_modules_b2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            2,
        )
        x_16 = l_self_modules_s1_modules_b2_modules_conv2_parameters_weight_ = None
        x_se_4 = x_17.mean((2, 3), keepdim=True)
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
        x_se_6 = torch.nn.functional.silu(x_se_5, inplace=True)
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
        x_18 = x_17 * sigmoid_1
        x_17 = sigmoid_1 = None
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_s1_modules_b2_modules_norm3_buffers_running_mean_,
            l_self_modules_s1_modules_b2_modules_norm3_buffers_running_var_,
            l_self_modules_s1_modules_b2_modules_norm3_parameters_weight_,
            l_self_modules_s1_modules_b2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = (
            l_self_modules_s1_modules_b2_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s1_modules_b2_modules_norm3_parameters_weight_
        ) = l_self_modules_s1_modules_b2_modules_norm3_parameters_bias_ = None
        x_20 = torch.nn.functional.silu(x_19, inplace=True)
        x_19 = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_s1_modules_b2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_s1_modules_b2_modules_conv3_parameters_weight_ = None
        x_22 = x_21 + x_13
        x_21 = x_13 = None
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_s2_modules_b1_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = (
            l_self_modules_s2_modules_b1_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b1_modules_norm1_parameters_bias_ = None
        x_24 = torch.nn.functional.silu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_s2_modules_b1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b1_modules_conv1_parameters_weight_ = None
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_s2_modules_b1_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = (
            l_self_modules_s2_modules_b1_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b1_modules_norm2_parameters_bias_ = None
        x_27 = torch.nn.functional.silu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_s2_modules_b1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            4,
        )
        x_27 = l_self_modules_s2_modules_b1_modules_conv2_parameters_weight_ = None
        x_se_8 = x_28.mean((2, 3), keepdim=True)
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
        x_se_10 = torch.nn.functional.silu(x_se_9, inplace=True)
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
        x_29 = x_28 * sigmoid_2
        x_28 = sigmoid_2 = None
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_s2_modules_b1_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b1_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b1_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = (
            l_self_modules_s2_modules_b1_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b1_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b1_modules_norm3_parameters_bias_ = None
        x_31 = torch.nn.functional.silu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_s2_modules_b1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_s2_modules_b1_modules_conv3_parameters_weight_ = None
        input_3 = torch._C._nn.avg_pool2d(x_24, 2, 2, 0, True, False, None)
        x_24 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_s2_modules_b1_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_3 = (
            l_self_modules_s2_modules_b1_modules_downsample_modules_1_parameters_weight_
        ) = None
        x_33 = x_32 + input_4
        x_32 = input_4 = None
        x_34 = torch.nn.functional.batch_norm(
            x_33,
            l_self_modules_s2_modules_b2_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_33 = (
            l_self_modules_s2_modules_b2_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b2_modules_norm1_parameters_bias_ = None
        x_35 = torch.nn.functional.silu(x_34, inplace=True)
        x_34 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_s2_modules_b2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b2_modules_conv1_parameters_weight_ = None
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_s2_modules_b2_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = (
            l_self_modules_s2_modules_b2_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b2_modules_norm2_parameters_bias_ = None
        x_38 = torch.nn.functional.silu(x_37, inplace=True)
        x_37 = None
        x_39 = torch.conv2d(
            x_38,
            l_self_modules_s2_modules_b2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_38 = l_self_modules_s2_modules_b2_modules_conv2_parameters_weight_ = None
        x_se_12 = x_39.mean((2, 3), keepdim=True)
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
        x_se_14 = torch.nn.functional.silu(x_se_13, inplace=True)
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
        x_40 = x_39 * sigmoid_3
        x_39 = sigmoid_3 = None
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_s2_modules_b2_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b2_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b2_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = (
            l_self_modules_s2_modules_b2_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b2_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b2_modules_norm3_parameters_bias_ = None
        x_42 = torch.nn.functional.silu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            x_42,
            l_self_modules_s2_modules_b2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_42 = l_self_modules_s2_modules_b2_modules_conv3_parameters_weight_ = None
        x_44 = x_43 + x_35
        x_43 = x_35 = None
        x_45 = torch.nn.functional.batch_norm(
            x_44,
            l_self_modules_s2_modules_b3_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_44 = (
            l_self_modules_s2_modules_b3_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b3_modules_norm1_parameters_bias_ = None
        x_46 = torch.nn.functional.silu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_s2_modules_b3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b3_modules_conv1_parameters_weight_ = None
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_s2_modules_b3_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = (
            l_self_modules_s2_modules_b3_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b3_modules_norm2_parameters_bias_ = None
        x_49 = torch.nn.functional.silu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_s2_modules_b3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_49 = l_self_modules_s2_modules_b3_modules_conv2_parameters_weight_ = None
        x_se_16 = x_50.mean((2, 3), keepdim=True)
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
        x_se_18 = torch.nn.functional.silu(x_se_17, inplace=True)
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
        x_51 = x_50 * sigmoid_4
        x_50 = sigmoid_4 = None
        x_52 = torch.nn.functional.batch_norm(
            x_51,
            l_self_modules_s2_modules_b3_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b3_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b3_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b3_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_51 = (
            l_self_modules_s2_modules_b3_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b3_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b3_modules_norm3_parameters_bias_ = None
        x_53 = torch.nn.functional.silu(x_52, inplace=True)
        x_52 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_s2_modules_b3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_s2_modules_b3_modules_conv3_parameters_weight_ = None
        x_55 = x_54 + x_46
        x_54 = x_46 = None
        x_56 = torch.nn.functional.batch_norm(
            x_55,
            l_self_modules_s2_modules_b4_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_55 = (
            l_self_modules_s2_modules_b4_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b4_modules_norm1_parameters_bias_ = None
        x_57 = torch.nn.functional.silu(x_56, inplace=True)
        x_56 = None
        x_58 = torch.conv2d(
            x_57,
            l_self_modules_s2_modules_b4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b4_modules_conv1_parameters_weight_ = None
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_s2_modules_b4_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = (
            l_self_modules_s2_modules_b4_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b4_modules_norm2_parameters_bias_ = None
        x_60 = torch.nn.functional.silu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            x_60,
            l_self_modules_s2_modules_b4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_60 = l_self_modules_s2_modules_b4_modules_conv2_parameters_weight_ = None
        x_se_20 = x_61.mean((2, 3), keepdim=True)
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
        x_se_22 = torch.nn.functional.silu(x_se_21, inplace=True)
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
        x_62 = x_61 * sigmoid_5
        x_61 = sigmoid_5 = None
        x_63 = torch.nn.functional.batch_norm(
            x_62,
            l_self_modules_s2_modules_b4_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b4_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b4_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b4_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_62 = (
            l_self_modules_s2_modules_b4_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b4_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b4_modules_norm3_parameters_bias_ = None
        x_64 = torch.nn.functional.silu(x_63, inplace=True)
        x_63 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_s2_modules_b4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_s2_modules_b4_modules_conv3_parameters_weight_ = None
        x_66 = x_65 + x_57
        x_65 = x_57 = None
        x_67 = torch.nn.functional.batch_norm(
            x_66,
            l_self_modules_s2_modules_b5_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_66 = (
            l_self_modules_s2_modules_b5_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b5_modules_norm1_parameters_bias_ = None
        x_68 = torch.nn.functional.silu(x_67, inplace=True)
        x_67 = None
        x_69 = torch.conv2d(
            x_68,
            l_self_modules_s2_modules_b5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b5_modules_conv1_parameters_weight_ = None
        x_70 = torch.nn.functional.batch_norm(
            x_69,
            l_self_modules_s2_modules_b5_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_69 = (
            l_self_modules_s2_modules_b5_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b5_modules_norm2_parameters_bias_ = None
        x_71 = torch.nn.functional.silu(x_70, inplace=True)
        x_70 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_s2_modules_b5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_71 = l_self_modules_s2_modules_b5_modules_conv2_parameters_weight_ = None
        x_se_24 = x_72.mean((2, 3), keepdim=True)
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
        x_se_26 = torch.nn.functional.silu(x_se_25, inplace=True)
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
        x_73 = x_72 * sigmoid_6
        x_72 = sigmoid_6 = None
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_s2_modules_b5_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b5_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b5_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b5_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = (
            l_self_modules_s2_modules_b5_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b5_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b5_modules_norm3_parameters_bias_ = None
        x_75 = torch.nn.functional.silu(x_74, inplace=True)
        x_74 = None
        x_76 = torch.conv2d(
            x_75,
            l_self_modules_s2_modules_b5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_s2_modules_b5_modules_conv3_parameters_weight_ = None
        x_77 = x_76 + x_68
        x_76 = x_68 = None
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_s2_modules_b6_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = (
            l_self_modules_s2_modules_b6_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b6_modules_norm1_parameters_bias_ = None
        x_79 = torch.nn.functional.silu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_s2_modules_b6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b6_modules_conv1_parameters_weight_ = None
        x_81 = torch.nn.functional.batch_norm(
            x_80,
            l_self_modules_s2_modules_b6_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_80 = (
            l_self_modules_s2_modules_b6_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b6_modules_norm2_parameters_bias_ = None
        x_82 = torch.nn.functional.silu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_s2_modules_b6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_82 = l_self_modules_s2_modules_b6_modules_conv2_parameters_weight_ = None
        x_se_28 = x_83.mean((2, 3), keepdim=True)
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
        x_se_30 = torch.nn.functional.silu(x_se_29, inplace=True)
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
        x_84 = x_83 * sigmoid_7
        x_83 = sigmoid_7 = None
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_s2_modules_b6_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b6_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b6_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b6_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = (
            l_self_modules_s2_modules_b6_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b6_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b6_modules_norm3_parameters_bias_ = None
        x_86 = torch.nn.functional.silu(x_85, inplace=True)
        x_85 = None
        x_87 = torch.conv2d(
            x_86,
            l_self_modules_s2_modules_b6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_86 = l_self_modules_s2_modules_b6_modules_conv3_parameters_weight_ = None
        x_88 = x_87 + x_79
        x_87 = x_79 = None
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_s2_modules_b7_modules_norm1_buffers_running_mean_,
            l_self_modules_s2_modules_b7_modules_norm1_buffers_running_var_,
            l_self_modules_s2_modules_b7_modules_norm1_parameters_weight_,
            l_self_modules_s2_modules_b7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = (
            l_self_modules_s2_modules_b7_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm1_parameters_weight_
        ) = l_self_modules_s2_modules_b7_modules_norm1_parameters_bias_ = None
        x_90 = torch.nn.functional.silu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_s2_modules_b7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s2_modules_b7_modules_conv1_parameters_weight_ = None
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_s2_modules_b7_modules_norm2_buffers_running_mean_,
            l_self_modules_s2_modules_b7_modules_norm2_buffers_running_var_,
            l_self_modules_s2_modules_b7_modules_norm2_parameters_weight_,
            l_self_modules_s2_modules_b7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = (
            l_self_modules_s2_modules_b7_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm2_parameters_weight_
        ) = l_self_modules_s2_modules_b7_modules_norm2_parameters_bias_ = None
        x_93 = torch.nn.functional.silu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_93,
            l_self_modules_s2_modules_b7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            4,
        )
        x_93 = l_self_modules_s2_modules_b7_modules_conv2_parameters_weight_ = None
        x_se_32 = x_94.mean((2, 3), keepdim=True)
        x_se_33 = torch.conv2d(
            x_se_32,
            l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_32 = (
            l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s2_modules_b7_modules_se_modules_fc1_parameters_bias_ = None
        x_se_34 = torch.nn.functional.silu(x_se_33, inplace=True)
        x_se_33 = None
        x_se_35 = torch.conv2d(
            x_se_34,
            l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_34 = (
            l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s2_modules_b7_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_8 = x_se_35.sigmoid()
        x_se_35 = None
        x_95 = x_94 * sigmoid_8
        x_94 = sigmoid_8 = None
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_s2_modules_b7_modules_norm3_buffers_running_mean_,
            l_self_modules_s2_modules_b7_modules_norm3_buffers_running_var_,
            l_self_modules_s2_modules_b7_modules_norm3_parameters_weight_,
            l_self_modules_s2_modules_b7_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = (
            l_self_modules_s2_modules_b7_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s2_modules_b7_modules_norm3_parameters_weight_
        ) = l_self_modules_s2_modules_b7_modules_norm3_parameters_bias_ = None
        x_97 = torch.nn.functional.silu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_s2_modules_b7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_s2_modules_b7_modules_conv3_parameters_weight_ = None
        x_99 = x_98 + x_90
        x_98 = x_90 = None
        x_100 = torch.nn.functional.batch_norm(
            x_99,
            l_self_modules_s3_modules_b1_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_99 = (
            l_self_modules_s3_modules_b1_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_norm1_parameters_bias_ = None
        x_101 = torch.nn.functional.silu(x_100, inplace=True)
        x_100 = None
        x_102 = torch.conv2d(
            x_101,
            l_self_modules_s3_modules_b1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b1_modules_conv1_parameters_weight_ = None
        x_103 = torch.nn.functional.batch_norm(
            x_102,
            l_self_modules_s3_modules_b1_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_102 = (
            l_self_modules_s3_modules_b1_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_norm2_parameters_bias_ = None
        x_104 = torch.nn.functional.silu(x_103, inplace=True)
        x_103 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_s3_modules_b1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        x_104 = l_self_modules_s3_modules_b1_modules_conv2_parameters_weight_ = None
        x_se_36 = x_105.mean((2, 3), keepdim=True)
        x_se_37 = torch.conv2d(
            x_se_36,
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_36 = (
            l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_38 = torch.nn.functional.silu(x_se_37, inplace=True)
        x_se_37 = None
        x_se_39 = torch.conv2d(
            x_se_38,
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_38 = (
            l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_9 = x_se_39.sigmoid()
        x_se_39 = None
        x_106 = x_105 * sigmoid_9
        x_105 = sigmoid_9 = None
        x_107 = torch.nn.functional.batch_norm(
            x_106,
            l_self_modules_s3_modules_b1_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b1_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b1_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_106 = (
            l_self_modules_s3_modules_b1_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b1_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b1_modules_norm3_parameters_bias_ = None
        x_108 = torch.nn.functional.silu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_s3_modules_b1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_108 = l_self_modules_s3_modules_b1_modules_conv3_parameters_weight_ = None
        input_5 = torch._C._nn.avg_pool2d(x_101, 2, 2, 0, True, False, None)
        x_101 = None
        input_6 = torch.conv2d(
            input_5,
            l_self_modules_s3_modules_b1_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_5 = (
            l_self_modules_s3_modules_b1_modules_downsample_modules_1_parameters_weight_
        ) = None
        x_110 = x_109 + input_6
        x_109 = input_6 = None
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_s3_modules_b2_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = (
            l_self_modules_s3_modules_b2_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_norm1_parameters_bias_ = None
        x_112 = torch.nn.functional.silu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_s3_modules_b2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b2_modules_conv1_parameters_weight_ = None
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_s3_modules_b2_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = (
            l_self_modules_s3_modules_b2_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_norm2_parameters_bias_ = None
        x_115 = torch.nn.functional.silu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_s3_modules_b2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_115 = l_self_modules_s3_modules_b2_modules_conv2_parameters_weight_ = None
        x_se_40 = x_116.mean((2, 3), keepdim=True)
        x_se_41 = torch.conv2d(
            x_se_40,
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_40 = (
            l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_42 = torch.nn.functional.silu(x_se_41, inplace=True)
        x_se_41 = None
        x_se_43 = torch.conv2d(
            x_se_42,
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_42 = (
            l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_10 = x_se_43.sigmoid()
        x_se_43 = None
        x_117 = x_116 * sigmoid_10
        x_116 = sigmoid_10 = None
        x_118 = torch.nn.functional.batch_norm(
            x_117,
            l_self_modules_s3_modules_b2_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b2_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b2_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_117 = (
            l_self_modules_s3_modules_b2_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b2_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b2_modules_norm3_parameters_bias_ = None
        x_119 = torch.nn.functional.silu(x_118, inplace=True)
        x_118 = None
        x_120 = torch.conv2d(
            x_119,
            l_self_modules_s3_modules_b2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_119 = l_self_modules_s3_modules_b2_modules_conv3_parameters_weight_ = None
        x_121 = x_120 + x_112
        x_120 = x_112 = None
        x_122 = torch.nn.functional.batch_norm(
            x_121,
            l_self_modules_s3_modules_b3_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_121 = (
            l_self_modules_s3_modules_b3_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_norm1_parameters_bias_ = None
        x_123 = torch.nn.functional.silu(x_122, inplace=True)
        x_122 = None
        x_124 = torch.conv2d(
            x_123,
            l_self_modules_s3_modules_b3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b3_modules_conv1_parameters_weight_ = None
        x_125 = torch.nn.functional.batch_norm(
            x_124,
            l_self_modules_s3_modules_b3_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_124 = (
            l_self_modules_s3_modules_b3_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_norm2_parameters_bias_ = None
        x_126 = torch.nn.functional.silu(x_125, inplace=True)
        x_125 = None
        x_127 = torch.conv2d(
            x_126,
            l_self_modules_s3_modules_b3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_126 = l_self_modules_s3_modules_b3_modules_conv2_parameters_weight_ = None
        x_se_44 = x_127.mean((2, 3), keepdim=True)
        x_se_45 = torch.conv2d(
            x_se_44,
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_44 = (
            l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_se_modules_fc1_parameters_bias_ = None
        x_se_46 = torch.nn.functional.silu(x_se_45, inplace=True)
        x_se_45 = None
        x_se_47 = torch.conv2d(
            x_se_46,
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_46 = (
            l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_11 = x_se_47.sigmoid()
        x_se_47 = None
        x_128 = x_127 * sigmoid_11
        x_127 = sigmoid_11 = None
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_s3_modules_b3_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b3_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b3_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b3_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = (
            l_self_modules_s3_modules_b3_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b3_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b3_modules_norm3_parameters_bias_ = None
        x_130 = torch.nn.functional.silu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_s3_modules_b3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_s3_modules_b3_modules_conv3_parameters_weight_ = None
        x_132 = x_131 + x_123
        x_131 = x_123 = None
        x_133 = torch.nn.functional.batch_norm(
            x_132,
            l_self_modules_s3_modules_b4_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_132 = (
            l_self_modules_s3_modules_b4_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_norm1_parameters_bias_ = None
        x_134 = torch.nn.functional.silu(x_133, inplace=True)
        x_133 = None
        x_135 = torch.conv2d(
            x_134,
            l_self_modules_s3_modules_b4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b4_modules_conv1_parameters_weight_ = None
        x_136 = torch.nn.functional.batch_norm(
            x_135,
            l_self_modules_s3_modules_b4_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_135 = (
            l_self_modules_s3_modules_b4_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_norm2_parameters_bias_ = None
        x_137 = torch.nn.functional.silu(x_136, inplace=True)
        x_136 = None
        x_138 = torch.conv2d(
            x_137,
            l_self_modules_s3_modules_b4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_137 = l_self_modules_s3_modules_b4_modules_conv2_parameters_weight_ = None
        x_se_48 = x_138.mean((2, 3), keepdim=True)
        x_se_49 = torch.conv2d(
            x_se_48,
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_48 = (
            l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_se_modules_fc1_parameters_bias_ = None
        x_se_50 = torch.nn.functional.silu(x_se_49, inplace=True)
        x_se_49 = None
        x_se_51 = torch.conv2d(
            x_se_50,
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_50 = (
            l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_12 = x_se_51.sigmoid()
        x_se_51 = None
        x_139 = x_138 * sigmoid_12
        x_138 = sigmoid_12 = None
        x_140 = torch.nn.functional.batch_norm(
            x_139,
            l_self_modules_s3_modules_b4_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b4_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b4_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b4_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_139 = (
            l_self_modules_s3_modules_b4_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b4_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b4_modules_norm3_parameters_bias_ = None
        x_141 = torch.nn.functional.silu(x_140, inplace=True)
        x_140 = None
        x_142 = torch.conv2d(
            x_141,
            l_self_modules_s3_modules_b4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_141 = l_self_modules_s3_modules_b4_modules_conv3_parameters_weight_ = None
        x_143 = x_142 + x_134
        x_142 = x_134 = None
        x_144 = torch.nn.functional.batch_norm(
            x_143,
            l_self_modules_s3_modules_b5_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_143 = (
            l_self_modules_s3_modules_b5_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_norm1_parameters_bias_ = None
        x_145 = torch.nn.functional.silu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.conv2d(
            x_145,
            l_self_modules_s3_modules_b5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b5_modules_conv1_parameters_weight_ = None
        x_147 = torch.nn.functional.batch_norm(
            x_146,
            l_self_modules_s3_modules_b5_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_146 = (
            l_self_modules_s3_modules_b5_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_norm2_parameters_bias_ = None
        x_148 = torch.nn.functional.silu(x_147, inplace=True)
        x_147 = None
        x_149 = torch.conv2d(
            x_148,
            l_self_modules_s3_modules_b5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_148 = l_self_modules_s3_modules_b5_modules_conv2_parameters_weight_ = None
        x_se_52 = x_149.mean((2, 3), keepdim=True)
        x_se_53 = torch.conv2d(
            x_se_52,
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_52 = (
            l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_se_modules_fc1_parameters_bias_ = None
        x_se_54 = torch.nn.functional.silu(x_se_53, inplace=True)
        x_se_53 = None
        x_se_55 = torch.conv2d(
            x_se_54,
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_54 = (
            l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_13 = x_se_55.sigmoid()
        x_se_55 = None
        x_150 = x_149 * sigmoid_13
        x_149 = sigmoid_13 = None
        x_151 = torch.nn.functional.batch_norm(
            x_150,
            l_self_modules_s3_modules_b5_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b5_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b5_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b5_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_150 = (
            l_self_modules_s3_modules_b5_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b5_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b5_modules_norm3_parameters_bias_ = None
        x_152 = torch.nn.functional.silu(x_151, inplace=True)
        x_151 = None
        x_153 = torch.conv2d(
            x_152,
            l_self_modules_s3_modules_b5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_152 = l_self_modules_s3_modules_b5_modules_conv3_parameters_weight_ = None
        x_154 = x_153 + x_145
        x_153 = x_145 = None
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_s3_modules_b6_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = (
            l_self_modules_s3_modules_b6_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_norm1_parameters_bias_ = None
        x_156 = torch.nn.functional.silu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_156,
            l_self_modules_s3_modules_b6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b6_modules_conv1_parameters_weight_ = None
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_s3_modules_b6_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = (
            l_self_modules_s3_modules_b6_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_norm2_parameters_bias_ = None
        x_159 = torch.nn.functional.silu(x_158, inplace=True)
        x_158 = None
        x_160 = torch.conv2d(
            x_159,
            l_self_modules_s3_modules_b6_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_159 = l_self_modules_s3_modules_b6_modules_conv2_parameters_weight_ = None
        x_se_56 = x_160.mean((2, 3), keepdim=True)
        x_se_57 = torch.conv2d(
            x_se_56,
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_56 = (
            l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_se_modules_fc1_parameters_bias_ = None
        x_se_58 = torch.nn.functional.silu(x_se_57, inplace=True)
        x_se_57 = None
        x_se_59 = torch.conv2d(
            x_se_58,
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_58 = (
            l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_14 = x_se_59.sigmoid()
        x_se_59 = None
        x_161 = x_160 * sigmoid_14
        x_160 = sigmoid_14 = None
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_s3_modules_b6_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b6_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b6_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b6_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = (
            l_self_modules_s3_modules_b6_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b6_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b6_modules_norm3_parameters_bias_ = None
        x_163 = torch.nn.functional.silu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_s3_modules_b6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_s3_modules_b6_modules_conv3_parameters_weight_ = None
        x_165 = x_164 + x_156
        x_164 = x_156 = None
        x_166 = torch.nn.functional.batch_norm(
            x_165,
            l_self_modules_s3_modules_b7_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_165 = (
            l_self_modules_s3_modules_b7_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_norm1_parameters_bias_ = None
        x_167 = torch.nn.functional.silu(x_166, inplace=True)
        x_166 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_s3_modules_b7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b7_modules_conv1_parameters_weight_ = None
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_s3_modules_b7_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = (
            l_self_modules_s3_modules_b7_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_norm2_parameters_bias_ = None
        x_170 = torch.nn.functional.silu(x_169, inplace=True)
        x_169 = None
        x_171 = torch.conv2d(
            x_170,
            l_self_modules_s3_modules_b7_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_170 = l_self_modules_s3_modules_b7_modules_conv2_parameters_weight_ = None
        x_se_60 = x_171.mean((2, 3), keepdim=True)
        x_se_61 = torch.conv2d(
            x_se_60,
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_60 = (
            l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_se_modules_fc1_parameters_bias_ = None
        x_se_62 = torch.nn.functional.silu(x_se_61, inplace=True)
        x_se_61 = None
        x_se_63 = torch.conv2d(
            x_se_62,
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_62 = (
            l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_15 = x_se_63.sigmoid()
        x_se_63 = None
        x_172 = x_171 * sigmoid_15
        x_171 = sigmoid_15 = None
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_s3_modules_b7_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b7_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b7_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b7_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = (
            l_self_modules_s3_modules_b7_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b7_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b7_modules_norm3_parameters_bias_ = None
        x_174 = torch.nn.functional.silu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_s3_modules_b7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_174 = l_self_modules_s3_modules_b7_modules_conv3_parameters_weight_ = None
        x_176 = x_175 + x_167
        x_175 = x_167 = None
        x_177 = torch.nn.functional.batch_norm(
            x_176,
            l_self_modules_s3_modules_b8_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_176 = (
            l_self_modules_s3_modules_b8_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_norm1_parameters_bias_ = None
        x_178 = torch.nn.functional.silu(x_177, inplace=True)
        x_177 = None
        x_179 = torch.conv2d(
            x_178,
            l_self_modules_s3_modules_b8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b8_modules_conv1_parameters_weight_ = None
        x_180 = torch.nn.functional.batch_norm(
            x_179,
            l_self_modules_s3_modules_b8_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_179 = (
            l_self_modules_s3_modules_b8_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_norm2_parameters_bias_ = None
        x_181 = torch.nn.functional.silu(x_180, inplace=True)
        x_180 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_s3_modules_b8_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_181 = l_self_modules_s3_modules_b8_modules_conv2_parameters_weight_ = None
        x_se_64 = x_182.mean((2, 3), keepdim=True)
        x_se_65 = torch.conv2d(
            x_se_64,
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_64 = (
            l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_se_modules_fc1_parameters_bias_ = None
        x_se_66 = torch.nn.functional.silu(x_se_65, inplace=True)
        x_se_65 = None
        x_se_67 = torch.conv2d(
            x_se_66,
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_66 = (
            l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_16 = x_se_67.sigmoid()
        x_se_67 = None
        x_183 = x_182 * sigmoid_16
        x_182 = sigmoid_16 = None
        x_184 = torch.nn.functional.batch_norm(
            x_183,
            l_self_modules_s3_modules_b8_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b8_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b8_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b8_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_183 = (
            l_self_modules_s3_modules_b8_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b8_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b8_modules_norm3_parameters_bias_ = None
        x_185 = torch.nn.functional.silu(x_184, inplace=True)
        x_184 = None
        x_186 = torch.conv2d(
            x_185,
            l_self_modules_s3_modules_b8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_185 = l_self_modules_s3_modules_b8_modules_conv3_parameters_weight_ = None
        x_187 = x_186 + x_178
        x_186 = x_178 = None
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_s3_modules_b9_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = (
            l_self_modules_s3_modules_b9_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_norm1_parameters_bias_ = None
        x_189 = torch.nn.functional.silu(x_188, inplace=True)
        x_188 = None
        x_190 = torch.conv2d(
            x_189,
            l_self_modules_s3_modules_b9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b9_modules_conv1_parameters_weight_ = None
        x_191 = torch.nn.functional.batch_norm(
            x_190,
            l_self_modules_s3_modules_b9_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_190 = (
            l_self_modules_s3_modules_b9_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_norm2_parameters_bias_ = None
        x_192 = torch.nn.functional.silu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_s3_modules_b9_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_192 = l_self_modules_s3_modules_b9_modules_conv2_parameters_weight_ = None
        x_se_68 = x_193.mean((2, 3), keepdim=True)
        x_se_69 = torch.conv2d(
            x_se_68,
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_68 = (
            l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_se_modules_fc1_parameters_bias_ = None
        x_se_70 = torch.nn.functional.silu(x_se_69, inplace=True)
        x_se_69 = None
        x_se_71 = torch.conv2d(
            x_se_70,
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_70 = (
            l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_17 = x_se_71.sigmoid()
        x_se_71 = None
        x_194 = x_193 * sigmoid_17
        x_193 = sigmoid_17 = None
        x_195 = torch.nn.functional.batch_norm(
            x_194,
            l_self_modules_s3_modules_b9_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b9_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b9_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b9_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_194 = (
            l_self_modules_s3_modules_b9_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b9_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b9_modules_norm3_parameters_bias_ = None
        x_196 = torch.nn.functional.silu(x_195, inplace=True)
        x_195 = None
        x_197 = torch.conv2d(
            x_196,
            l_self_modules_s3_modules_b9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_196 = l_self_modules_s3_modules_b9_modules_conv3_parameters_weight_ = None
        x_198 = x_197 + x_189
        x_197 = x_189 = None
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_s3_modules_b10_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = (
            l_self_modules_s3_modules_b10_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_norm1_parameters_bias_ = None
        x_200 = torch.nn.functional.silu(x_199, inplace=True)
        x_199 = None
        x_201 = torch.conv2d(
            x_200,
            l_self_modules_s3_modules_b10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b10_modules_conv1_parameters_weight_ = None
        x_202 = torch.nn.functional.batch_norm(
            x_201,
            l_self_modules_s3_modules_b10_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_201 = (
            l_self_modules_s3_modules_b10_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_norm2_parameters_bias_ = None
        x_203 = torch.nn.functional.silu(x_202, inplace=True)
        x_202 = None
        x_204 = torch.conv2d(
            x_203,
            l_self_modules_s3_modules_b10_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_203 = l_self_modules_s3_modules_b10_modules_conv2_parameters_weight_ = None
        x_se_72 = x_204.mean((2, 3), keepdim=True)
        x_se_73 = torch.conv2d(
            x_se_72,
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_72 = (
            l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_se_modules_fc1_parameters_bias_ = None
        x_se_74 = torch.nn.functional.silu(x_se_73, inplace=True)
        x_se_73 = None
        x_se_75 = torch.conv2d(
            x_se_74,
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_74 = (
            l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_18 = x_se_75.sigmoid()
        x_se_75 = None
        x_205 = x_204 * sigmoid_18
        x_204 = sigmoid_18 = None
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_s3_modules_b10_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b10_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b10_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b10_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_205 = (
            l_self_modules_s3_modules_b10_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b10_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b10_modules_norm3_parameters_bias_ = None
        x_207 = torch.nn.functional.silu(x_206, inplace=True)
        x_206 = None
        x_208 = torch.conv2d(
            x_207,
            l_self_modules_s3_modules_b10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_207 = l_self_modules_s3_modules_b10_modules_conv3_parameters_weight_ = None
        x_209 = x_208 + x_200
        x_208 = x_200 = None
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_s3_modules_b11_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = (
            l_self_modules_s3_modules_b11_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_norm1_parameters_bias_ = None
        x_211 = torch.nn.functional.silu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_s3_modules_b11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b11_modules_conv1_parameters_weight_ = None
        x_213 = torch.nn.functional.batch_norm(
            x_212,
            l_self_modules_s3_modules_b11_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_212 = (
            l_self_modules_s3_modules_b11_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_norm2_parameters_bias_ = None
        x_214 = torch.nn.functional.silu(x_213, inplace=True)
        x_213 = None
        x_215 = torch.conv2d(
            x_214,
            l_self_modules_s3_modules_b11_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_214 = l_self_modules_s3_modules_b11_modules_conv2_parameters_weight_ = None
        x_se_76 = x_215.mean((2, 3), keepdim=True)
        x_se_77 = torch.conv2d(
            x_se_76,
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_76 = (
            l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_se_modules_fc1_parameters_bias_ = None
        x_se_78 = torch.nn.functional.silu(x_se_77, inplace=True)
        x_se_77 = None
        x_se_79 = torch.conv2d(
            x_se_78,
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_78 = (
            l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_19 = x_se_79.sigmoid()
        x_se_79 = None
        x_216 = x_215 * sigmoid_19
        x_215 = sigmoid_19 = None
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_s3_modules_b11_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b11_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b11_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b11_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = (
            l_self_modules_s3_modules_b11_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b11_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b11_modules_norm3_parameters_bias_ = None
        x_218 = torch.nn.functional.silu(x_217, inplace=True)
        x_217 = None
        x_219 = torch.conv2d(
            x_218,
            l_self_modules_s3_modules_b11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_218 = l_self_modules_s3_modules_b11_modules_conv3_parameters_weight_ = None
        x_220 = x_219 + x_211
        x_219 = x_211 = None
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_s3_modules_b12_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = (
            l_self_modules_s3_modules_b12_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_norm1_parameters_bias_ = None
        x_222 = torch.nn.functional.silu(x_221, inplace=True)
        x_221 = None
        x_223 = torch.conv2d(
            x_222,
            l_self_modules_s3_modules_b12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b12_modules_conv1_parameters_weight_ = None
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_s3_modules_b12_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = (
            l_self_modules_s3_modules_b12_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_norm2_parameters_bias_ = None
        x_225 = torch.nn.functional.silu(x_224, inplace=True)
        x_224 = None
        x_226 = torch.conv2d(
            x_225,
            l_self_modules_s3_modules_b12_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_225 = l_self_modules_s3_modules_b12_modules_conv2_parameters_weight_ = None
        x_se_80 = x_226.mean((2, 3), keepdim=True)
        x_se_81 = torch.conv2d(
            x_se_80,
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_80 = (
            l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_se_modules_fc1_parameters_bias_ = None
        x_se_82 = torch.nn.functional.silu(x_se_81, inplace=True)
        x_se_81 = None
        x_se_83 = torch.conv2d(
            x_se_82,
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_82 = (
            l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_20 = x_se_83.sigmoid()
        x_se_83 = None
        x_227 = x_226 * sigmoid_20
        x_226 = sigmoid_20 = None
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_s3_modules_b12_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b12_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b12_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b12_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = (
            l_self_modules_s3_modules_b12_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b12_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b12_modules_norm3_parameters_bias_ = None
        x_229 = torch.nn.functional.silu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_s3_modules_b12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_s3_modules_b12_modules_conv3_parameters_weight_ = None
        x_231 = x_230 + x_222
        x_230 = x_222 = None
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_s3_modules_b13_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b13_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b13_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b13_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = (
            l_self_modules_s3_modules_b13_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b13_modules_norm1_parameters_bias_ = None
        x_233 = torch.nn.functional.silu(x_232, inplace=True)
        x_232 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_s3_modules_b13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b13_modules_conv1_parameters_weight_ = None
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_s3_modules_b13_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b13_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b13_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b13_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = (
            l_self_modules_s3_modules_b13_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b13_modules_norm2_parameters_bias_ = None
        x_236 = torch.nn.functional.silu(x_235, inplace=True)
        x_235 = None
        x_237 = torch.conv2d(
            x_236,
            l_self_modules_s3_modules_b13_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_236 = l_self_modules_s3_modules_b13_modules_conv2_parameters_weight_ = None
        x_se_84 = x_237.mean((2, 3), keepdim=True)
        x_se_85 = torch.conv2d(
            x_se_84,
            l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_84 = (
            l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b13_modules_se_modules_fc1_parameters_bias_ = None
        x_se_86 = torch.nn.functional.silu(x_se_85, inplace=True)
        x_se_85 = None
        x_se_87 = torch.conv2d(
            x_se_86,
            l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_86 = (
            l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b13_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_21 = x_se_87.sigmoid()
        x_se_87 = None
        x_238 = x_237 * sigmoid_21
        x_237 = sigmoid_21 = None
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_s3_modules_b13_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b13_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b13_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b13_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = (
            l_self_modules_s3_modules_b13_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b13_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b13_modules_norm3_parameters_bias_ = None
        x_240 = torch.nn.functional.silu(x_239, inplace=True)
        x_239 = None
        x_241 = torch.conv2d(
            x_240,
            l_self_modules_s3_modules_b13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_240 = l_self_modules_s3_modules_b13_modules_conv3_parameters_weight_ = None
        x_242 = x_241 + x_233
        x_241 = x_233 = None
        x_243 = torch.nn.functional.batch_norm(
            x_242,
            l_self_modules_s3_modules_b14_modules_norm1_buffers_running_mean_,
            l_self_modules_s3_modules_b14_modules_norm1_buffers_running_var_,
            l_self_modules_s3_modules_b14_modules_norm1_parameters_weight_,
            l_self_modules_s3_modules_b14_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_242 = (
            l_self_modules_s3_modules_b14_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm1_parameters_weight_
        ) = l_self_modules_s3_modules_b14_modules_norm1_parameters_bias_ = None
        x_244 = torch.nn.functional.silu(x_243, inplace=True)
        x_243 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_s3_modules_b14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s3_modules_b14_modules_conv1_parameters_weight_ = None
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_s3_modules_b14_modules_norm2_buffers_running_mean_,
            l_self_modules_s3_modules_b14_modules_norm2_buffers_running_var_,
            l_self_modules_s3_modules_b14_modules_norm2_parameters_weight_,
            l_self_modules_s3_modules_b14_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = (
            l_self_modules_s3_modules_b14_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm2_parameters_weight_
        ) = l_self_modules_s3_modules_b14_modules_norm2_parameters_bias_ = None
        x_247 = torch.nn.functional.silu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_s3_modules_b14_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        x_247 = l_self_modules_s3_modules_b14_modules_conv2_parameters_weight_ = None
        x_se_88 = x_248.mean((2, 3), keepdim=True)
        x_se_89 = torch.conv2d(
            x_se_88,
            l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_88 = (
            l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s3_modules_b14_modules_se_modules_fc1_parameters_bias_ = None
        x_se_90 = torch.nn.functional.silu(x_se_89, inplace=True)
        x_se_89 = None
        x_se_91 = torch.conv2d(
            x_se_90,
            l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_90 = (
            l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s3_modules_b14_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_22 = x_se_91.sigmoid()
        x_se_91 = None
        x_249 = x_248 * sigmoid_22
        x_248 = sigmoid_22 = None
        x_250 = torch.nn.functional.batch_norm(
            x_249,
            l_self_modules_s3_modules_b14_modules_norm3_buffers_running_mean_,
            l_self_modules_s3_modules_b14_modules_norm3_buffers_running_var_,
            l_self_modules_s3_modules_b14_modules_norm3_parameters_weight_,
            l_self_modules_s3_modules_b14_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_249 = (
            l_self_modules_s3_modules_b14_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s3_modules_b14_modules_norm3_parameters_weight_
        ) = l_self_modules_s3_modules_b14_modules_norm3_parameters_bias_ = None
        x_251 = torch.nn.functional.silu(x_250, inplace=True)
        x_250 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_s3_modules_b14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_s3_modules_b14_modules_conv3_parameters_weight_ = None
        x_253 = x_252 + x_244
        x_252 = x_244 = None
        x_254 = torch.nn.functional.batch_norm(
            x_253,
            l_self_modules_s4_modules_b1_modules_norm1_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_norm1_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_norm1_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_253 = (
            l_self_modules_s4_modules_b1_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm1_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_norm1_parameters_bias_ = None
        x_255 = torch.nn.functional.silu(x_254, inplace=True)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_s4_modules_b1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b1_modules_conv1_parameters_weight_ = None
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_s4_modules_b1_modules_norm2_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_norm2_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_norm2_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = (
            l_self_modules_s4_modules_b1_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm2_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_norm2_parameters_bias_ = None
        x_258 = torch.nn.functional.silu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_s4_modules_b1_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            18,
        )
        x_258 = l_self_modules_s4_modules_b1_modules_conv2_parameters_weight_ = None
        x_se_92 = x_259.mean((2, 3), keepdim=True)
        x_se_93 = torch.conv2d(
            x_se_92,
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_92 = (
            l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_se_modules_fc1_parameters_bias_ = None
        x_se_94 = torch.nn.functional.silu(x_se_93, inplace=True)
        x_se_93 = None
        x_se_95 = torch.conv2d(
            x_se_94,
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_94 = (
            l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_23 = x_se_95.sigmoid()
        x_se_95 = None
        x_260 = x_259 * sigmoid_23
        x_259 = sigmoid_23 = None
        x_261 = torch.nn.functional.batch_norm(
            x_260,
            l_self_modules_s4_modules_b1_modules_norm3_buffers_running_mean_,
            l_self_modules_s4_modules_b1_modules_norm3_buffers_running_var_,
            l_self_modules_s4_modules_b1_modules_norm3_parameters_weight_,
            l_self_modules_s4_modules_b1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_260 = (
            l_self_modules_s4_modules_b1_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b1_modules_norm3_parameters_weight_
        ) = l_self_modules_s4_modules_b1_modules_norm3_parameters_bias_ = None
        x_262 = torch.nn.functional.silu(x_261, inplace=True)
        x_261 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_s4_modules_b1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_s4_modules_b1_modules_conv3_parameters_weight_ = None
        input_7 = torch._C._nn.avg_pool2d(x_255, 2, 2, 0, True, False, None)
        x_255 = None
        input_8 = torch.conv2d(
            input_7,
            l_self_modules_s4_modules_b1_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_7 = (
            l_self_modules_s4_modules_b1_modules_downsample_modules_1_parameters_weight_
        ) = None
        x_264 = x_263 + input_8
        x_263 = input_8 = None
        x_265 = torch.nn.functional.batch_norm(
            x_264,
            l_self_modules_s4_modules_b2_modules_norm1_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_norm1_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_norm1_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_264 = (
            l_self_modules_s4_modules_b2_modules_norm1_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm1_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm1_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_norm1_parameters_bias_ = None
        x_266 = torch.nn.functional.silu(x_265, inplace=True)
        x_265 = None
        x_267 = torch.conv2d(
            x_266,
            l_self_modules_s4_modules_b2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_s4_modules_b2_modules_conv1_parameters_weight_ = None
        x_268 = torch.nn.functional.batch_norm(
            x_267,
            l_self_modules_s4_modules_b2_modules_norm2_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_norm2_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_norm2_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_267 = (
            l_self_modules_s4_modules_b2_modules_norm2_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm2_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm2_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_norm2_parameters_bias_ = None
        x_269 = torch.nn.functional.silu(x_268, inplace=True)
        x_268 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_s4_modules_b2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            18,
        )
        x_269 = l_self_modules_s4_modules_b2_modules_conv2_parameters_weight_ = None
        x_se_96 = x_270.mean((2, 3), keepdim=True)
        x_se_97 = torch.conv2d(
            x_se_96,
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_96 = (
            l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_se_modules_fc1_parameters_bias_ = None
        x_se_98 = torch.nn.functional.silu(x_se_97, inplace=True)
        x_se_97 = None
        x_se_99 = torch.conv2d(
            x_se_98,
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_se_98 = (
            l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_se_modules_fc2_parameters_bias_ = None
        sigmoid_24 = x_se_99.sigmoid()
        x_se_99 = None
        x_271 = x_270 * sigmoid_24
        x_270 = sigmoid_24 = None
        x_272 = torch.nn.functional.batch_norm(
            x_271,
            l_self_modules_s4_modules_b2_modules_norm3_buffers_running_mean_,
            l_self_modules_s4_modules_b2_modules_norm3_buffers_running_var_,
            l_self_modules_s4_modules_b2_modules_norm3_parameters_weight_,
            l_self_modules_s4_modules_b2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_271 = (
            l_self_modules_s4_modules_b2_modules_norm3_buffers_running_mean_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm3_buffers_running_var_
        ) = (
            l_self_modules_s4_modules_b2_modules_norm3_parameters_weight_
        ) = l_self_modules_s4_modules_b2_modules_norm3_parameters_bias_ = None
        x_273 = torch.nn.functional.silu(x_272, inplace=True)
        x_272 = None
        x_274 = torch.conv2d(
            x_273,
            l_self_modules_s4_modules_b2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_273 = l_self_modules_s4_modules_b2_modules_conv3_parameters_weight_ = None
        x_275 = x_274 + x_266
        x_274 = x_266 = None
        x_276 = torch.nn.functional.silu(x_275, inplace=False)
        x_275 = None
        x_277 = torch.nn.functional.adaptive_avg_pool2d(x_276, 1)
        x_276 = None
        x_278 = x_277.flatten(1, -1)
        x_277 = None
        x_279 = torch.nn.functional.dropout(x_278, 0.0, False, False)
        x_278 = None
        x_280 = torch._C._nn.linear(
            x_279,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
        )
        x_279 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        return (x_280,)
