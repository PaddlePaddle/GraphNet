import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_fc_parameters_bias_: torch.nn.parameter.Parameter,
    ):
        l_self_modules_conv1_parameters_weight_ = (
            L_self_modules_conv1_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_bn1_buffers_running_mean_ = (
            L_self_modules_bn1_buffers_running_mean_
        )
        l_self_modules_bn1_buffers_running_var_ = (
            L_self_modules_bn1_buffers_running_var_
        )
        l_self_modules_bn1_parameters_weight_ = L_self_modules_bn1_parameters_weight_
        l_self_modules_bn1_parameters_bias_ = L_self_modules_bn1_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv3_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv3_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_bias_
        )
        l_self_modules_fc_parameters_weight_ = L_self_modules_fc_parameters_weight_
        l_self_modules_fc_parameters_bias_ = L_self_modules_fc_parameters_bias_
        x = torch.conv2d(
            l_x_,
            l_self_modules_conv1_parameters_weight_,
            None,
            (2, 2),
            (3, 3),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_parameters_weight_ = None
        x_1 = torch.nn.functional.batch_norm(
            x,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_2 = torch.nn.functional.relu(x_1, inplace=True)
        x_1 = None
        x_3 = torch.nn.functional.max_pool2d(
            x_2, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_2 = None
        x_4 = torch.conv2d(
            x_3,
            l_self_modules_layer1_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            x_6,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        x_10 = torch.conv2d(
            x_6,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_6 = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_11 = torch.nn.functional.batch_norm(
            x_10,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_10 = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_12 = torch.nn.functional.relu(x_11, inplace=True)
        x_11 = None
        stack = torch.stack([x_9, x_12], dim=1)
        x_9 = x_12 = None
        sym_sum = torch.sym_sum([-1, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        sum_1 = stack.sum(1)
        x_13 = sum_1.mean((2, 3), keepdim=True)
        sum_1 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_15 = torch.nn.functional.batch_norm(
            x_14,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_14 = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_16 = torch.nn.functional.relu(x_15, inplace=True)
        x_15 = None
        x_17 = torch.conv2d(
            x_16,
            l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_16 = l_self_modules_layer1_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_18 = x_17.view(1, 2, 128, 1, 1)
        x_17 = None
        x_19 = torch.softmax(x_18, dim=1)
        x_18 = None
        mul = stack * x_19
        stack = x_19 = None
        x_20 = torch.sum(mul, dim=1)
        mul = None
        x_21 = torch.conv2d(
            x_20,
            l_self_modules_layer1_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_20 = l_self_modules_layer1_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_22 = torch.nn.functional.batch_norm(
            x_21,
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_21 = l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv3_modules_bn_parameters_bias_
        ) = None
        input_1 = torch.conv2d(
            x_3,
            l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_3 = l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_22 += input_2
        x_23 = x_22
        x_22 = input_2 = None
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            x_24,
            l_self_modules_layer1_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        x_28 = torch.conv2d(
            x_27,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_29 = torch.nn.functional.batch_norm(
            x_28,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_28 = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_30 = torch.nn.functional.relu(x_29, inplace=True)
        x_29 = None
        x_31 = torch.conv2d(
            x_27,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_27 = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_32 = torch.nn.functional.batch_norm(
            x_31,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_31 = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_33 = torch.nn.functional.relu(x_32, inplace=True)
        x_32 = None
        stack_1 = torch.stack([x_30, x_33], dim=1)
        x_30 = x_33 = None
        sum_3 = stack_1.sum(1)
        x_34 = sum_3.mean((2, 3), keepdim=True)
        sum_3 = None
        x_35 = torch.conv2d(
            x_34,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_34 = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_36 = torch.nn.functional.batch_norm(
            x_35,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_35 = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_37 = torch.nn.functional.relu(x_36, inplace=True)
        x_36 = None
        x_38 = torch.conv2d(
            x_37,
            l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_37 = l_self_modules_layer1_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_39 = x_38.view(1, 2, 128, 1, 1)
        x_38 = None
        x_40 = torch.softmax(x_39, dim=1)
        x_39 = None
        mul_1 = stack_1 * x_40
        stack_1 = x_40 = None
        x_41 = torch.sum(mul_1, dim=1)
        mul_1 = None
        x_42 = torch.conv2d(
            x_41,
            l_self_modules_layer1_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_41 = l_self_modules_layer1_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_43 = torch.nn.functional.batch_norm(
            x_42,
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_42 = l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_43 += x_24
        x_44 = x_43
        x_43 = x_24 = None
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        x_46 = torch.conv2d(
            x_45,
            l_self_modules_layer1_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_47 = torch.nn.functional.batch_norm(
            x_46,
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_46 = l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_48 = torch.nn.functional.relu(x_47, inplace=True)
        x_47 = None
        x_49 = torch.conv2d(
            x_48,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_50 = torch.nn.functional.batch_norm(
            x_49,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_49 = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_51 = torch.nn.functional.relu(x_50, inplace=True)
        x_50 = None
        x_52 = torch.conv2d(
            x_48,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_48 = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_53 = torch.nn.functional.batch_norm(
            x_52,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_52 = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_54 = torch.nn.functional.relu(x_53, inplace=True)
        x_53 = None
        stack_2 = torch.stack([x_51, x_54], dim=1)
        x_51 = x_54 = None
        sum_5 = stack_2.sum(1)
        x_55 = sum_5.mean((2, 3), keepdim=True)
        sum_5 = None
        x_56 = torch.conv2d(
            x_55,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_55 = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_57 = torch.nn.functional.batch_norm(
            x_56,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_56 = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_58 = torch.nn.functional.relu(x_57, inplace=True)
        x_57 = None
        x_59 = torch.conv2d(
            x_58,
            l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_58 = l_self_modules_layer1_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_60 = x_59.view(1, 2, 128, 1, 1)
        x_59 = None
        x_61 = torch.softmax(x_60, dim=1)
        x_60 = None
        mul_2 = stack_2 * x_61
        stack_2 = x_61 = None
        x_62 = torch.sum(mul_2, dim=1)
        mul_2 = None
        x_63 = torch.conv2d(
            x_62,
            l_self_modules_layer1_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_62 = l_self_modules_layer1_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_64 = torch.nn.functional.batch_norm(
            x_63,
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_63 = l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_64 += x_45
        x_65 = x_64
        x_64 = x_45 = None
        x_66 = torch.nn.functional.relu(x_65, inplace=True)
        x_65 = None
        x_67 = torch.conv2d(
            x_66,
            l_self_modules_layer2_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_68 = torch.nn.functional.batch_norm(
            x_67,
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_67 = l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_69 = torch.nn.functional.relu(x_68, inplace=True)
        x_68 = None
        x_70 = torch.conv2d(
            x_69,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_71 = torch.nn.functional.batch_norm(
            x_70,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_70 = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_72 = torch.nn.functional.relu(x_71, inplace=True)
        x_71 = None
        x_73 = torch.conv2d(
            x_69,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            32,
        )
        x_69 = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_74 = torch.nn.functional.batch_norm(
            x_73,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_73 = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        stack_3 = torch.stack([x_72, x_75], dim=1)
        x_72 = x_75 = None
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        sum_7 = stack_3.sum(1)
        x_76 = sum_7.mean((2, 3), keepdim=True)
        sum_7 = None
        x_77 = torch.conv2d(
            x_76,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_76 = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_78 = torch.nn.functional.batch_norm(
            x_77,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_77 = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_79 = torch.nn.functional.relu(x_78, inplace=True)
        x_78 = None
        x_80 = torch.conv2d(
            x_79,
            l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_79 = l_self_modules_layer2_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_81 = x_80.view(1, 2, 256, 1, 1)
        x_80 = None
        x_82 = torch.softmax(x_81, dim=1)
        x_81 = None
        mul_3 = stack_3 * x_82
        stack_3 = x_82 = None
        x_83 = torch.sum(mul_3, dim=1)
        mul_3 = None
        x_84 = torch.conv2d(
            x_83,
            l_self_modules_layer2_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_83 = l_self_modules_layer2_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_85 = torch.nn.functional.batch_norm(
            x_84,
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_84 = l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv3_modules_bn_parameters_bias_
        ) = None
        input_3 = torch.conv2d(
            x_66,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_66 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_85 += input_4
        x_86 = x_85
        x_85 = input_4 = None
        x_87 = torch.nn.functional.relu(x_86, inplace=True)
        x_86 = None
        x_88 = torch.conv2d(
            x_87,
            l_self_modules_layer2_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_89 = torch.nn.functional.batch_norm(
            x_88,
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_88 = l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_90 = torch.nn.functional.relu(x_89, inplace=True)
        x_89 = None
        x_91 = torch.conv2d(
            x_90,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_92 = torch.nn.functional.batch_norm(
            x_91,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_91 = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        x_94 = torch.conv2d(
            x_90,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_90 = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        stack_4 = torch.stack([x_93, x_96], dim=1)
        x_93 = x_96 = None
        sum_9 = stack_4.sum(1)
        x_97 = sum_9.mean((2, 3), keepdim=True)
        sum_9 = None
        x_98 = torch.conv2d(
            x_97,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_97 = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_99 = torch.nn.functional.batch_norm(
            x_98,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_98 = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_100 = torch.nn.functional.relu(x_99, inplace=True)
        x_99 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_layer2_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_102 = x_101.view(1, 2, 256, 1, 1)
        x_101 = None
        x_103 = torch.softmax(x_102, dim=1)
        x_102 = None
        mul_4 = stack_4 * x_103
        stack_4 = x_103 = None
        x_104 = torch.sum(mul_4, dim=1)
        mul_4 = None
        x_105 = torch.conv2d(
            x_104,
            l_self_modules_layer2_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_104 = l_self_modules_layer2_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_106 = torch.nn.functional.batch_norm(
            x_105,
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_105 = l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_106 += x_87
        x_107 = x_106
        x_106 = x_87 = None
        x_108 = torch.nn.functional.relu(x_107, inplace=True)
        x_107 = None
        x_109 = torch.conv2d(
            x_108,
            l_self_modules_layer2_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_110 = torch.nn.functional.batch_norm(
            x_109,
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_109 = l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        x_112 = torch.conv2d(
            x_111,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            x_111,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_111 = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        stack_5 = torch.stack([x_114, x_117], dim=1)
        x_114 = x_117 = None
        sum_11 = stack_5.sum(1)
        x_118 = sum_11.mean((2, 3), keepdim=True)
        sum_11 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_layer2_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_123 = x_122.view(1, 2, 256, 1, 1)
        x_122 = None
        x_124 = torch.softmax(x_123, dim=1)
        x_123 = None
        mul_5 = stack_5 * x_124
        stack_5 = x_124 = None
        x_125 = torch.sum(mul_5, dim=1)
        mul_5 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_layer2_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_layer2_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_127 += x_108
        x_128 = x_127
        x_127 = x_108 = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        x_130 = torch.conv2d(
            x_129,
            l_self_modules_layer2_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            x_132,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        x_136 = torch.conv2d(
            x_132,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_132 = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_137 = torch.nn.functional.batch_norm(
            x_136,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_136 = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_138 = torch.nn.functional.relu(x_137, inplace=True)
        x_137 = None
        stack_6 = torch.stack([x_135, x_138], dim=1)
        x_135 = x_138 = None
        sum_13 = stack_6.sum(1)
        x_139 = sum_13.mean((2, 3), keepdim=True)
        sum_13 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_141 = torch.nn.functional.batch_norm(
            x_140,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_140 = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_142 = torch.nn.functional.relu(x_141, inplace=True)
        x_141 = None
        x_143 = torch.conv2d(
            x_142,
            l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_142 = l_self_modules_layer2_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_144 = x_143.view(1, 2, 256, 1, 1)
        x_143 = None
        x_145 = torch.softmax(x_144, dim=1)
        x_144 = None
        mul_6 = stack_6 * x_145
        stack_6 = x_145 = None
        x_146 = torch.sum(mul_6, dim=1)
        mul_6 = None
        x_147 = torch.conv2d(
            x_146,
            l_self_modules_layer2_modules_3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_146 = l_self_modules_layer2_modules_3_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_148 = torch.nn.functional.batch_norm(
            x_147,
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_147 = l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_148 += x_129
        x_149 = x_148
        x_148 = x_129 = None
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            x_150,
            l_self_modules_layer3_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        x_154 = torch.conv2d(
            x_153,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_155 = torch.nn.functional.batch_norm(
            x_154,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_154 = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_156 = torch.nn.functional.relu(x_155, inplace=True)
        x_155 = None
        x_157 = torch.conv2d(
            x_153,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            32,
        )
        x_153 = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_158 = torch.nn.functional.batch_norm(
            x_157,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_157 = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_159 = torch.nn.functional.relu(x_158, inplace=True)
        x_158 = None
        stack_7 = torch.stack([x_156, x_159], dim=1)
        x_156 = x_159 = None
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        sum_15 = stack_7.sum(1)
        x_160 = sum_15.mean((2, 3), keepdim=True)
        sum_15 = None
        x_161 = torch.conv2d(
            x_160,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_160 = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_162 = torch.nn.functional.batch_norm(
            x_161,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_161 = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_163 = torch.nn.functional.relu(x_162, inplace=True)
        x_162 = None
        x_164 = torch.conv2d(
            x_163,
            l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_163 = l_self_modules_layer3_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_165 = x_164.view(1, 2, 512, 1, 1)
        x_164 = None
        x_166 = torch.softmax(x_165, dim=1)
        x_165 = None
        mul_7 = stack_7 * x_166
        stack_7 = x_166 = None
        x_167 = torch.sum(mul_7, dim=1)
        mul_7 = None
        x_168 = torch.conv2d(
            x_167,
            l_self_modules_layer3_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_167 = l_self_modules_layer3_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_169 = torch.nn.functional.batch_norm(
            x_168,
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_168 = l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv3_modules_bn_parameters_bias_
        ) = None
        input_5 = torch.conv2d(
            x_150,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_150 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_169 += input_6
        x_170 = x_169
        x_169 = input_6 = None
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        x_172 = torch.conv2d(
            x_171,
            l_self_modules_layer3_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_173 = torch.nn.functional.batch_norm(
            x_172,
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_172 = l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_174 = torch.nn.functional.relu(x_173, inplace=True)
        x_173 = None
        x_175 = torch.conv2d(
            x_174,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_176 = torch.nn.functional.batch_norm(
            x_175,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_175 = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_177 = torch.nn.functional.relu(x_176, inplace=True)
        x_176 = None
        x_178 = torch.conv2d(
            x_174,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_174 = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_179 = torch.nn.functional.batch_norm(
            x_178,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_178 = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_180 = torch.nn.functional.relu(x_179, inplace=True)
        x_179 = None
        stack_8 = torch.stack([x_177, x_180], dim=1)
        x_177 = x_180 = None
        sum_17 = stack_8.sum(1)
        x_181 = sum_17.mean((2, 3), keepdim=True)
        sum_17 = None
        x_182 = torch.conv2d(
            x_181,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_181 = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_183 = torch.nn.functional.batch_norm(
            x_182,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_182 = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_184 = torch.nn.functional.relu(x_183, inplace=True)
        x_183 = None
        x_185 = torch.conv2d(
            x_184,
            l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_184 = l_self_modules_layer3_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_186 = x_185.view(1, 2, 512, 1, 1)
        x_185 = None
        x_187 = torch.softmax(x_186, dim=1)
        x_186 = None
        mul_8 = stack_8 * x_187
        stack_8 = x_187 = None
        x_188 = torch.sum(mul_8, dim=1)
        mul_8 = None
        x_189 = torch.conv2d(
            x_188,
            l_self_modules_layer3_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_188 = l_self_modules_layer3_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_190 = torch.nn.functional.batch_norm(
            x_189,
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_189 = l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_190 += x_171
        x_191 = x_190
        x_190 = x_171 = None
        x_192 = torch.nn.functional.relu(x_191, inplace=True)
        x_191 = None
        x_193 = torch.conv2d(
            x_192,
            l_self_modules_layer3_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_194 = torch.nn.functional.batch_norm(
            x_193,
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_193 = l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_195 = torch.nn.functional.relu(x_194, inplace=True)
        x_194 = None
        x_196 = torch.conv2d(
            x_195,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_197 = torch.nn.functional.batch_norm(
            x_196,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_196 = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_198 = torch.nn.functional.relu(x_197, inplace=True)
        x_197 = None
        x_199 = torch.conv2d(
            x_195,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_195 = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_200 = torch.nn.functional.batch_norm(
            x_199,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_199 = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        stack_9 = torch.stack([x_198, x_201], dim=1)
        x_198 = x_201 = None
        sum_19 = stack_9.sum(1)
        x_202 = sum_19.mean((2, 3), keepdim=True)
        sum_19 = None
        x_203 = torch.conv2d(
            x_202,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_202 = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_204 = torch.nn.functional.batch_norm(
            x_203,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_203 = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_205 = torch.nn.functional.relu(x_204, inplace=True)
        x_204 = None
        x_206 = torch.conv2d(
            x_205,
            l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_205 = l_self_modules_layer3_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_207 = x_206.view(1, 2, 512, 1, 1)
        x_206 = None
        x_208 = torch.softmax(x_207, dim=1)
        x_207 = None
        mul_9 = stack_9 * x_208
        stack_9 = x_208 = None
        x_209 = torch.sum(mul_9, dim=1)
        mul_9 = None
        x_210 = torch.conv2d(
            x_209,
            l_self_modules_layer3_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_209 = l_self_modules_layer3_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_211 = torch.nn.functional.batch_norm(
            x_210,
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_210 = l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_211 += x_192
        x_212 = x_211
        x_211 = x_192 = None
        x_213 = torch.nn.functional.relu(x_212, inplace=True)
        x_212 = None
        x_214 = torch.conv2d(
            x_213,
            l_self_modules_layer3_modules_3_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_215 = torch.nn.functional.batch_norm(
            x_214,
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_214 = l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_216 = torch.nn.functional.relu(x_215, inplace=True)
        x_215 = None
        x_217 = torch.conv2d(
            x_216,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_218 = torch.nn.functional.batch_norm(
            x_217,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_217 = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_219 = torch.nn.functional.relu(x_218, inplace=True)
        x_218 = None
        x_220 = torch.conv2d(
            x_216,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_216 = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        stack_10 = torch.stack([x_219, x_222], dim=1)
        x_219 = x_222 = None
        sum_21 = stack_10.sum(1)
        x_223 = sum_21.mean((2, 3), keepdim=True)
        sum_21 = None
        x_224 = torch.conv2d(
            x_223,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_223 = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_225 = torch.nn.functional.batch_norm(
            x_224,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_224 = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_226 = torch.nn.functional.relu(x_225, inplace=True)
        x_225 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_layer3_modules_3_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_228 = x_227.view(1, 2, 512, 1, 1)
        x_227 = None
        x_229 = torch.softmax(x_228, dim=1)
        x_228 = None
        mul_10 = stack_10 * x_229
        stack_10 = x_229 = None
        x_230 = torch.sum(mul_10, dim=1)
        mul_10 = None
        x_231 = torch.conv2d(
            x_230,
            l_self_modules_layer3_modules_3_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_230 = l_self_modules_layer3_modules_3_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_232 = torch.nn.functional.batch_norm(
            x_231,
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_231 = l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_232 += x_213
        x_233 = x_232
        x_232 = x_213 = None
        x_234 = torch.nn.functional.relu(x_233, inplace=True)
        x_233 = None
        x_235 = torch.conv2d(
            x_234,
            l_self_modules_layer3_modules_4_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_236 = torch.nn.functional.batch_norm(
            x_235,
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_235 = l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        x_238 = torch.conv2d(
            x_237,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_240 = torch.nn.functional.relu(x_239, inplace=True)
        x_239 = None
        x_241 = torch.conv2d(
            x_237,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_237 = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        stack_11 = torch.stack([x_240, x_243], dim=1)
        x_240 = x_243 = None
        sum_23 = stack_11.sum(1)
        x_244 = sum_23.mean((2, 3), keepdim=True)
        sum_23 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_layer3_modules_4_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_249 = x_248.view(1, 2, 512, 1, 1)
        x_248 = None
        x_250 = torch.softmax(x_249, dim=1)
        x_249 = None
        mul_11 = stack_11 * x_250
        stack_11 = x_250 = None
        x_251 = torch.sum(mul_11, dim=1)
        mul_11 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_layer3_modules_4_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_layer3_modules_4_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_253 += x_234
        x_254 = x_253
        x_253 = x_234 = None
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        x_256 = torch.conv2d(
            x_255,
            l_self_modules_layer3_modules_5_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_258 = torch.nn.functional.relu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            x_258,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_261 = torch.nn.functional.relu(x_260, inplace=True)
        x_260 = None
        x_262 = torch.conv2d(
            x_258,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_258 = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_263 = torch.nn.functional.batch_norm(
            x_262,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_262 = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_264 = torch.nn.functional.relu(x_263, inplace=True)
        x_263 = None
        stack_12 = torch.stack([x_261, x_264], dim=1)
        x_261 = x_264 = None
        sum_25 = stack_12.sum(1)
        x_265 = sum_25.mean((2, 3), keepdim=True)
        sum_25 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_267 = torch.nn.functional.batch_norm(
            x_266,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_266 = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_268 = torch.nn.functional.relu(x_267, inplace=True)
        x_267 = None
        x_269 = torch.conv2d(
            x_268,
            l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_268 = l_self_modules_layer3_modules_5_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_270 = x_269.view(1, 2, 512, 1, 1)
        x_269 = None
        x_271 = torch.softmax(x_270, dim=1)
        x_270 = None
        mul_12 = stack_12 * x_271
        stack_12 = x_271 = None
        x_272 = torch.sum(mul_12, dim=1)
        mul_12 = None
        x_273 = torch.conv2d(
            x_272,
            l_self_modules_layer3_modules_5_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_272 = l_self_modules_layer3_modules_5_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_274 = torch.nn.functional.batch_norm(
            x_273,
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_273 = l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_274 += x_255
        x_275 = x_274
        x_274 = x_255 = None
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            x_276,
            l_self_modules_layer4_modules_0_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        x_280 = torch.conv2d(
            x_279,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_281 = torch.nn.functional.batch_norm(
            x_280,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_280 = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_282 = torch.nn.functional.relu(x_281, inplace=True)
        x_281 = None
        x_283 = torch.conv2d(
            x_279,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            32,
        )
        x_279 = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_284 = torch.nn.functional.batch_norm(
            x_283,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_283 = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_285 = torch.nn.functional.relu(x_284, inplace=True)
        x_284 = None
        stack_13 = torch.stack([x_282, x_285], dim=1)
        x_282 = x_285 = None
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        sum_27 = stack_13.sum(1)
        x_286 = sum_27.mean((2, 3), keepdim=True)
        sum_27 = None
        x_287 = torch.conv2d(
            x_286,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_286 = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_288 = torch.nn.functional.batch_norm(
            x_287,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_287 = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_289 = torch.nn.functional.relu(x_288, inplace=True)
        x_288 = None
        x_290 = torch.conv2d(
            x_289,
            l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_289 = l_self_modules_layer4_modules_0_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_291 = x_290.view(1, 2, 1024, 1, 1)
        x_290 = None
        x_292 = torch.softmax(x_291, dim=1)
        x_291 = None
        mul_13 = stack_13 * x_292
        stack_13 = x_292 = None
        x_293 = torch.sum(mul_13, dim=1)
        mul_13 = None
        x_294 = torch.conv2d(
            x_293,
            l_self_modules_layer4_modules_0_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_293 = l_self_modules_layer4_modules_0_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_295 = torch.nn.functional.batch_norm(
            x_294,
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_294 = l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv3_modules_bn_parameters_bias_
        ) = None
        input_7 = torch.conv2d(
            x_276,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_276 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_8 = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_295 += input_8
        x_296 = x_295
        x_295 = input_8 = None
        x_297 = torch.nn.functional.relu(x_296, inplace=True)
        x_296 = None
        x_298 = torch.conv2d(
            x_297,
            l_self_modules_layer4_modules_1_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_299 = torch.nn.functional.batch_norm(
            x_298,
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_298 = l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_300 = torch.nn.functional.relu(x_299, inplace=True)
        x_299 = None
        x_301 = torch.conv2d(
            x_300,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_302 = torch.nn.functional.batch_norm(
            x_301,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_301 = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_303 = torch.nn.functional.relu(x_302, inplace=True)
        x_302 = None
        x_304 = torch.conv2d(
            x_300,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_300 = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_305 = torch.nn.functional.batch_norm(
            x_304,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_304 = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_306 = torch.nn.functional.relu(x_305, inplace=True)
        x_305 = None
        stack_14 = torch.stack([x_303, x_306], dim=1)
        x_303 = x_306 = None
        sum_29 = stack_14.sum(1)
        x_307 = sum_29.mean((2, 3), keepdim=True)
        sum_29 = None
        x_308 = torch.conv2d(
            x_307,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_307 = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_309 = torch.nn.functional.batch_norm(
            x_308,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_308 = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_310 = torch.nn.functional.relu(x_309, inplace=True)
        x_309 = None
        x_311 = torch.conv2d(
            x_310,
            l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_310 = l_self_modules_layer4_modules_1_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_312 = x_311.view(1, 2, 1024, 1, 1)
        x_311 = None
        x_313 = torch.softmax(x_312, dim=1)
        x_312 = None
        mul_14 = stack_14 * x_313
        stack_14 = x_313 = None
        x_314 = torch.sum(mul_14, dim=1)
        mul_14 = None
        x_315 = torch.conv2d(
            x_314,
            l_self_modules_layer4_modules_1_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_314 = l_self_modules_layer4_modules_1_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_316 = torch.nn.functional.batch_norm(
            x_315,
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_315 = l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_316 += x_297
        x_317 = x_316
        x_316 = x_297 = None
        x_318 = torch.nn.functional.relu(x_317, inplace=True)
        x_317 = None
        x_319 = torch.conv2d(
            x_318,
            l_self_modules_layer4_modules_2_modules_conv1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_modules_conv_parameters_weight_ = (
            None
        )
        x_320 = torch.nn.functional.batch_norm(
            x_319,
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_319 = l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv1_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv1_modules_bn_parameters_bias_
        ) = None
        x_321 = torch.nn.functional.relu(x_320, inplace=True)
        x_320 = None
        x_322 = torch.conv2d(
            x_321,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            32,
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_conv_parameters_weight_ = (
            None
        )
        x_323 = torch.nn.functional.batch_norm(
            x_322,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_322 = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_324 = torch.nn.functional.relu(x_323, inplace=True)
        x_323 = None
        x_325 = torch.conv2d(
            x_321,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            32,
        )
        x_321 = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_326 = torch.nn.functional.batch_norm(
            x_325,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_325 = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv2_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_327 = torch.nn.functional.relu(x_326, inplace=True)
        x_326 = None
        stack_15 = torch.stack([x_324, x_327], dim=1)
        x_324 = x_327 = None
        sum_31 = stack_15.sum(1)
        x_328 = sum_31.mean((2, 3), keepdim=True)
        sum_31 = None
        x_329 = torch.conv2d(
            x_328,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_328 = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_330 = torch.nn.functional.batch_norm(
            x_329,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_329 = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_bn_parameters_bias_ = (None)
        x_331 = torch.nn.functional.relu(x_330, inplace=True)
        x_330 = None
        x_332 = torch.conv2d(
            x_331,
            l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_331 = l_self_modules_layer4_modules_2_modules_conv2_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_333 = x_332.view(1, 2, 1024, 1, 1)
        x_332 = None
        x_334 = torch.softmax(x_333, dim=1)
        x_333 = None
        mul_15 = stack_15 * x_334
        stack_15 = x_334 = None
        x_335 = torch.sum(mul_15, dim=1)
        mul_15 = None
        x_336 = torch.conv2d(
            x_335,
            l_self_modules_layer4_modules_2_modules_conv3_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_335 = l_self_modules_layer4_modules_2_modules_conv3_modules_conv_parameters_weight_ = (None)
        x_337 = torch.nn.functional.batch_norm(
            x_336,
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_336 = l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv3_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv3_modules_bn_parameters_bias_
        ) = None
        x_337 += x_318
        x_338 = x_337
        x_337 = x_318 = None
        x_339 = torch.nn.functional.relu(x_338, inplace=True)
        x_338 = None
        x_340 = torch.nn.functional.adaptive_avg_pool2d(x_339, 1)
        x_339 = None
        x_341 = x_340.flatten(1, -1)
        x_340 = None
        x_342 = torch._C._nn.linear(
            x_341,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_341 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_342,)
