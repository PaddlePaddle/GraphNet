import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_stem_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_mean_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_var_: torch.Tensor,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_buffers_running_mean_: torch.Tensor,
        L_self_modules_norm_buffers_running_var_: torch.Tensor,
        L_self_modules_norm_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_norm_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_head_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_stem_modules_conv_parameters_weight_ = (
            L_self_modules_stem_modules_conv_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_mean_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_var_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_weight_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_bias_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_bias_
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_parameters_weight_ = L_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_parameters_weight_
        l_self_modules_norm_buffers_running_mean_ = (
            L_self_modules_norm_buffers_running_mean_
        )
        l_self_modules_norm_buffers_running_var_ = (
            L_self_modules_norm_buffers_running_var_
        )
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_norm_parameters_bias_ = L_self_modules_norm_parameters_bias_
        l_self_modules_head_modules_fc_parameters_weight_ = (
            L_self_modules_head_modules_fc_parameters_weight_
        )
        l_self_modules_head_modules_fc_parameters_bias_ = (
            L_self_modules_head_modules_fc_parameters_bias_
        )
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_stem_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_stem_modules_conv_parameters_weight_ = None
        input_2 = torch.nn.functional.max_pool2d(
            input_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        input_1 = None
        x = torch.nn.functional.batch_norm(
            input_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_2 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        shortcut = torch.conv2d(
            x_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = (
            None
        )
        x_2 = torch.conv2d(
            x_1,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_1 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv1_parameters_weight_ = (None)
        x_3 = torch.nn.functional.batch_norm(
            x_2,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_2 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_4 = torch.nn.functional.relu(x_3, inplace=True)
        x_3 = None
        x_5 = torch.conv2d(
            x_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_4 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv2_parameters_weight_ = (None)
        x_6 = torch.nn.functional.batch_norm(
            x_5,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_5 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_norm3_parameters_bias_ = (None)
        x_7 = torch.nn.functional.relu(x_6, inplace=True)
        x_6 = None
        x_8 = torch.conv2d(
            x_7,
            l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_7 = l_self_modules_stages_modules_0_modules_blocks_modules_0_modules_conv3_parameters_weight_ = (None)
        input_3 = x_8 + shortcut
        x_8 = shortcut = None
        x_9 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_10 = torch.nn.functional.relu(x_9, inplace=True)
        x_9 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv1_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv2_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_stages_modules_0_modules_blocks_modules_1_modules_conv3_parameters_weight_ = (None)
        input_4 = x_17 + input_3
        x_17 = input_3 = None
        x_18 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_19 = torch.nn.functional.relu(x_18, inplace=True)
        x_18 = None
        x_20 = torch.conv2d(
            x_19,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_19 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv1_parameters_weight_ = (None)
        x_21 = torch.nn.functional.batch_norm(
            x_20,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_20 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_22 = torch.nn.functional.relu(x_21, inplace=True)
        x_21 = None
        x_23 = torch.conv2d(
            x_22,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_22 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv2_parameters_weight_ = (None)
        x_24 = torch.nn.functional.batch_norm(
            x_23,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_23 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_norm3_parameters_bias_ = (None)
        x_25 = torch.nn.functional.relu(x_24, inplace=True)
        x_24 = None
        x_26 = torch.conv2d(
            x_25,
            l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_25 = l_self_modules_stages_modules_0_modules_blocks_modules_2_modules_conv3_parameters_weight_ = (None)
        input_5 = x_26 + input_4
        x_26 = input_4 = None
        x_27 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_28 = torch.nn.functional.relu(x_27, inplace=True)
        x_27 = None
        shortcut_1 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = (
            None
        )
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv1_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv2_parameters_weight_ = (None)
        x_33 = torch.nn.functional.batch_norm(
            x_32,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_32 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_norm3_parameters_bias_ = (None)
        x_34 = torch.nn.functional.relu(x_33, inplace=True)
        x_33 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_stages_modules_1_modules_blocks_modules_0_modules_conv3_parameters_weight_ = (None)
        input_6 = x_35 + shortcut_1
        x_35 = shortcut_1 = None
        x_36 = torch.nn.functional.batch_norm(
            input_6,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv1_parameters_weight_ = (None)
        x_39 = torch.nn.functional.batch_norm(
            x_38,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_38 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_40 = torch.nn.functional.relu(x_39, inplace=True)
        x_39 = None
        x_41 = torch.conv2d(
            x_40,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_40 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv2_parameters_weight_ = (None)
        x_42 = torch.nn.functional.batch_norm(
            x_41,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_41 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_43 = torch.nn.functional.relu(x_42, inplace=True)
        x_42 = None
        x_44 = torch.conv2d(
            x_43,
            l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_43 = l_self_modules_stages_modules_1_modules_blocks_modules_1_modules_conv3_parameters_weight_ = (None)
        input_7 = x_44 + input_6
        x_44 = input_6 = None
        x_45 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_46 = torch.nn.functional.relu(x_45, inplace=True)
        x_45 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv1_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv2_parameters_weight_ = (None)
        x_51 = torch.nn.functional.batch_norm(
            x_50,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_50 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_norm3_parameters_bias_ = (None)
        x_52 = torch.nn.functional.relu(x_51, inplace=True)
        x_51 = None
        x_53 = torch.conv2d(
            x_52,
            l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_52 = l_self_modules_stages_modules_1_modules_blocks_modules_2_modules_conv3_parameters_weight_ = (None)
        input_8 = x_53 + input_7
        x_53 = input_7 = None
        x_54 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_55 = torch.nn.functional.relu(x_54, inplace=True)
        x_54 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv1_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv2_parameters_weight_ = (None)
        x_60 = torch.nn.functional.batch_norm(
            x_59,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_59 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_norm3_parameters_bias_ = (None)
        x_61 = torch.nn.functional.relu(x_60, inplace=True)
        x_60 = None
        x_62 = torch.conv2d(
            x_61,
            l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_61 = l_self_modules_stages_modules_1_modules_blocks_modules_3_modules_conv3_parameters_weight_ = (None)
        input_9 = x_62 + input_8
        x_62 = input_8 = None
        x_63 = torch.nn.functional.batch_norm(
            input_9,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_9 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_64 = torch.nn.functional.relu(x_63, inplace=True)
        x_63 = None
        shortcut_2 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = (
            None
        )
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv1_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv2_parameters_weight_ = (None)
        x_69 = torch.nn.functional.batch_norm(
            x_68,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_68 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_norm3_parameters_bias_ = (None)
        x_70 = torch.nn.functional.relu(x_69, inplace=True)
        x_69 = None
        x_71 = torch.conv2d(
            x_70,
            l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_70 = l_self_modules_stages_modules_2_modules_blocks_modules_0_modules_conv3_parameters_weight_ = (None)
        input_10 = x_71 + shortcut_2
        x_71 = shortcut_2 = None
        x_72 = torch.nn.functional.batch_norm(
            input_10,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_73 = torch.nn.functional.relu(x_72, inplace=True)
        x_72 = None
        x_74 = torch.conv2d(
            x_73,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_73 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv1_parameters_weight_ = (None)
        x_75 = torch.nn.functional.batch_norm(
            x_74,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_74 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_76 = torch.nn.functional.relu(x_75, inplace=True)
        x_75 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv2_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_stages_modules_2_modules_blocks_modules_1_modules_conv3_parameters_weight_ = (None)
        input_11 = x_80 + input_10
        x_80 = input_10 = None
        x_81 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_82 = torch.nn.functional.relu(x_81, inplace=True)
        x_81 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv1_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv2_parameters_weight_ = (None)
        x_87 = torch.nn.functional.batch_norm(
            x_86,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_86 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_norm3_parameters_bias_ = (None)
        x_88 = torch.nn.functional.relu(x_87, inplace=True)
        x_87 = None
        x_89 = torch.conv2d(
            x_88,
            l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_88 = l_self_modules_stages_modules_2_modules_blocks_modules_2_modules_conv3_parameters_weight_ = (None)
        input_12 = x_89 + input_11
        x_89 = input_11 = None
        x_90 = torch.nn.functional.batch_norm(
            input_12,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm1_parameters_bias_ = (None)
        x_91 = torch.nn.functional.relu(x_90, inplace=True)
        x_90 = None
        x_92 = torch.conv2d(
            x_91,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_91 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv1_parameters_weight_ = (None)
        x_93 = torch.nn.functional.batch_norm(
            x_92,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_92 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm2_parameters_bias_ = (None)
        x_94 = torch.nn.functional.relu(x_93, inplace=True)
        x_93 = None
        x_95 = torch.conv2d(
            x_94,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_94 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv2_parameters_weight_ = (None)
        x_96 = torch.nn.functional.batch_norm(
            x_95,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_95 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_norm3_parameters_bias_ = (None)
        x_97 = torch.nn.functional.relu(x_96, inplace=True)
        x_96 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_stages_modules_2_modules_blocks_modules_3_modules_conv3_parameters_weight_ = (None)
        input_13 = x_98 + input_12
        x_98 = input_12 = None
        x_99 = torch.nn.functional.batch_norm(
            input_13,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm1_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv1_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm2_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv2_parameters_weight_ = (None)
        x_105 = torch.nn.functional.batch_norm(
            x_104,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_104 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_norm3_parameters_bias_ = (None)
        x_106 = torch.nn.functional.relu(x_105, inplace=True)
        x_105 = None
        x_107 = torch.conv2d(
            x_106,
            l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_106 = l_self_modules_stages_modules_2_modules_blocks_modules_4_modules_conv3_parameters_weight_ = (None)
        input_14 = x_107 + input_13
        x_107 = input_13 = None
        x_108 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm1_parameters_bias_ = (None)
        x_109 = torch.nn.functional.relu(x_108, inplace=True)
        x_108 = None
        x_110 = torch.conv2d(
            x_109,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_109 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv1_parameters_weight_ = (None)
        x_111 = torch.nn.functional.batch_norm(
            x_110,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_110 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm2_parameters_bias_ = (None)
        x_112 = torch.nn.functional.relu(x_111, inplace=True)
        x_111 = None
        x_113 = torch.conv2d(
            x_112,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_112 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv2_parameters_weight_ = (None)
        x_114 = torch.nn.functional.batch_norm(
            x_113,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_113 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_norm3_parameters_bias_ = (None)
        x_115 = torch.nn.functional.relu(x_114, inplace=True)
        x_114 = None
        x_116 = torch.conv2d(
            x_115,
            l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_115 = l_self_modules_stages_modules_2_modules_blocks_modules_5_modules_conv3_parameters_weight_ = (None)
        input_15 = x_116 + input_14
        x_116 = input_14 = None
        x_117 = torch.nn.functional.batch_norm(
            input_15,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_15 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm1_parameters_bias_ = (None)
        x_118 = torch.nn.functional.relu(x_117, inplace=True)
        x_117 = None
        shortcut_3 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_downsample_modules_conv_parameters_weight_ = (
            None
        )
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv1_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm2_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv2_parameters_weight_ = (None)
        x_123 = torch.nn.functional.batch_norm(
            x_122,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_122 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_norm3_parameters_bias_ = (None)
        x_124 = torch.nn.functional.relu(x_123, inplace=True)
        x_123 = None
        x_125 = torch.conv2d(
            x_124,
            l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_124 = l_self_modules_stages_modules_3_modules_blocks_modules_0_modules_conv3_parameters_weight_ = (None)
        input_16 = x_125 + shortcut_3
        x_125 = shortcut_3 = None
        x_126 = torch.nn.functional.batch_norm(
            input_16,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm1_parameters_bias_ = (None)
        x_127 = torch.nn.functional.relu(x_126, inplace=True)
        x_126 = None
        x_128 = torch.conv2d(
            x_127,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_127 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv1_parameters_weight_ = (None)
        x_129 = torch.nn.functional.batch_norm(
            x_128,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_128 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm2_parameters_bias_ = (None)
        x_130 = torch.nn.functional.relu(x_129, inplace=True)
        x_129 = None
        x_131 = torch.conv2d(
            x_130,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_130 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv2_parameters_weight_ = (None)
        x_132 = torch.nn.functional.batch_norm(
            x_131,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_131 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_norm3_parameters_bias_ = (None)
        x_133 = torch.nn.functional.relu(x_132, inplace=True)
        x_132 = None
        x_134 = torch.conv2d(
            x_133,
            l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_133 = l_self_modules_stages_modules_3_modules_blocks_modules_1_modules_conv3_parameters_weight_ = (None)
        input_17 = x_134 + input_16
        x_134 = input_16 = None
        x_135 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm1_parameters_bias_ = (None)
        x_136 = torch.nn.functional.relu(x_135, inplace=True)
        x_135 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv1_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm2_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv2_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_mean_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_var_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_weight_,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_mean_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_buffers_running_var_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_weight_ = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_norm3_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_stages_modules_3_modules_blocks_modules_2_modules_conv3_parameters_weight_ = (None)
        input_18 = x_143 + input_17
        x_143 = input_17 = None
        x_144 = torch.nn.functional.batch_norm(
            input_18,
            l_self_modules_norm_buffers_running_mean_,
            l_self_modules_norm_buffers_running_var_,
            l_self_modules_norm_parameters_weight_,
            l_self_modules_norm_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_18 = (
            l_self_modules_norm_buffers_running_mean_
        ) = (
            l_self_modules_norm_buffers_running_var_
        ) = (
            l_self_modules_norm_parameters_weight_
        ) = l_self_modules_norm_parameters_bias_ = None
        x_145 = torch.nn.functional.relu(x_144, inplace=True)
        x_144 = None
        x_146 = torch.nn.functional.adaptive_avg_pool2d(x_145, 1)
        x_145 = None
        x_147 = torch.nn.functional.dropout(x_146, 0.0, False, False)
        x_146 = None
        x_148 = torch.conv2d(
            x_147,
            l_self_modules_head_modules_fc_parameters_weight_,
            l_self_modules_head_modules_fc_parameters_bias_,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_147 = (
            l_self_modules_head_modules_fc_parameters_weight_
        ) = l_self_modules_head_modules_fc_parameters_bias_ = None
        x_149 = x_148.flatten(1, -1)
        x_148 = None
        return (x_149,)
