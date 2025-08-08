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
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_
        l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = L_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_mean_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_var_ = L_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_var_
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_bias_
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
        split = torch.functional.split(x_3, 32, 1)
        getitem = split[0]
        getitem_1 = split[1]
        split = None
        x_4 = torch.conv2d(
            getitem,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_5 = torch.nn.functional.batch_norm(
            x_4,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_4 = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_6 = torch.nn.functional.relu(x_5, inplace=True)
        x_5 = None
        x_7 = torch.conv2d(
            getitem_1,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_1 = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_8 = torch.nn.functional.batch_norm(
            x_7,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_7 = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_9 = torch.nn.functional.relu(x_8, inplace=True)
        x_8 = None
        stack = torch.stack([x_6, x_9], dim=1)
        x_6 = x_9 = None
        sym_sum = torch.sym_sum([-1, s1])
        s1 = None
        floordiv = sym_sum // 4
        sym_sum_1 = torch.sym_sum([1, floordiv])
        floordiv = sym_sum_1 = None
        sum_1 = stack.sum(1)
        x_10 = sum_1.mean((2, 3), keepdim=True)
        sum_1 = None
        x_11 = torch.conv2d(
            x_10,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_10 = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_12 = torch.nn.functional.batch_norm(
            x_11,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_11 = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_13 = torch.nn.functional.relu(x_12, inplace=True)
        x_12 = None
        x_14 = torch.conv2d(
            x_13,
            l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_13 = l_self_modules_layer1_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_15 = x_14.view(1, 2, 64, 1, 1)
        x_14 = None
        x_16 = torch.softmax(x_15, dim=1)
        x_15 = None
        mul = stack * x_16
        stack = x_16 = None
        x_17 = torch.sum(mul, dim=1)
        mul = None
        x_18 = torch.conv2d(
            x_17,
            l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_17 = l_self_modules_layer1_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_19 = torch.nn.functional.batch_norm(
            x_18,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_18 = l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_19 += x_3
        x_20 = x_19
        x_19 = x_3 = None
        x_21 = torch.nn.functional.relu(x_20, inplace=True)
        x_20 = None
        split_1 = torch.functional.split(x_21, 32, 1)
        getitem_7 = split_1[0]
        getitem_8 = split_1[1]
        split_1 = None
        x_22 = torch.conv2d(
            getitem_7,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_7 = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_23 = torch.nn.functional.batch_norm(
            x_22,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_22 = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_24 = torch.nn.functional.relu(x_23, inplace=True)
        x_23 = None
        x_25 = torch.conv2d(
            getitem_8,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_8 = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_26 = torch.nn.functional.batch_norm(
            x_25,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_25 = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_27 = torch.nn.functional.relu(x_26, inplace=True)
        x_26 = None
        stack_1 = torch.stack([x_24, x_27], dim=1)
        x_24 = x_27 = None
        sum_3 = stack_1.sum(1)
        x_28 = sum_3.mean((2, 3), keepdim=True)
        sum_3 = None
        x_29 = torch.conv2d(
            x_28,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_28 = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_30 = torch.nn.functional.batch_norm(
            x_29,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_29 = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_31 = torch.nn.functional.relu(x_30, inplace=True)
        x_30 = None
        x_32 = torch.conv2d(
            x_31,
            l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_31 = l_self_modules_layer1_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_33 = x_32.view(1, 2, 64, 1, 1)
        x_32 = None
        x_34 = torch.softmax(x_33, dim=1)
        x_33 = None
        mul_1 = stack_1 * x_34
        stack_1 = x_34 = None
        x_35 = torch.sum(mul_1, dim=1)
        mul_1 = None
        x_36 = torch.conv2d(
            x_35,
            l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_35 = l_self_modules_layer1_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_37 = torch.nn.functional.batch_norm(
            x_36,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_36 = l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_37 += x_21
        x_38 = x_37
        x_37 = x_21 = None
        x_39 = torch.nn.functional.relu(x_38, inplace=True)
        x_38 = None
        split_2 = torch.functional.split(x_39, 32, 1)
        getitem_14 = split_2[0]
        getitem_15 = split_2[1]
        split_2 = None
        x_40 = torch.conv2d(
            getitem_14,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_14 = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            getitem_15,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_15 = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        stack_2 = torch.stack([x_42, x_45], dim=1)
        x_42 = x_45 = None
        sum_5 = stack_2.sum(1)
        x_46 = sum_5.mean((2, 3), keepdim=True)
        sum_5 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_layer1_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_51 = x_50.view(1, 2, 64, 1, 1)
        x_50 = None
        x_52 = torch.softmax(x_51, dim=1)
        x_51 = None
        mul_2 = stack_2 * x_52
        stack_2 = x_52 = None
        x_53 = torch.sum(mul_2, dim=1)
        mul_2 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_layer1_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer1_modules_2_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_55 += x_39
        x_56 = x_55
        x_55 = x_39 = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        split_3 = torch.functional.split(x_57, 32, 1)
        getitem_21 = split_3[0]
        getitem_22 = split_3[1]
        split_3 = None
        x_58 = torch.conv2d(
            getitem_21,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_21 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            getitem_22,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_22 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        stack_3 = torch.stack([x_60, x_63], dim=1)
        x_60 = x_63 = None
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        sum_7 = stack_3.sum(1)
        x_64 = sum_7.mean((2, 3), keepdim=True)
        sum_7 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_69 = x_68.view(1, 2, 128, 1, 1)
        x_68 = None
        x_70 = torch.softmax(x_69, dim=1)
        x_69 = None
        mul_3 = stack_3 * x_70
        stack_3 = x_70 = None
        x_71 = torch.sum(mul_3, dim=1)
        mul_3 = None
        x_72 = torch.conv2d(
            x_71,
            l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_1 = torch.conv2d(
            x_57,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_57 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_73 += input_2
        x_74 = x_73
        x_73 = input_2 = None
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        split_4 = torch.functional.split(x_75, 64, 1)
        getitem_28 = split_4[0]
        getitem_29 = split_4[1]
        split_4 = None
        x_76 = torch.conv2d(
            getitem_28,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_28 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            getitem_29,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_29 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        stack_4 = torch.stack([x_78, x_81], dim=1)
        x_78 = x_81 = None
        sum_9 = stack_4.sum(1)
        x_82 = sum_9.mean((2, 3), keepdim=True)
        sum_9 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_87 = x_86.view(1, 2, 128, 1, 1)
        x_86 = None
        x_88 = torch.softmax(x_87, dim=1)
        x_87 = None
        mul_4 = stack_4 * x_88
        stack_4 = x_88 = None
        x_89 = torch.sum(mul_4, dim=1)
        mul_4 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_91 += x_75
        x_92 = x_91
        x_91 = x_75 = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        split_5 = torch.functional.split(x_93, 64, 1)
        getitem_35 = split_5[0]
        getitem_36 = split_5[1]
        split_5 = None
        x_94 = torch.conv2d(
            getitem_35,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_35 = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            getitem_36,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_36 = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        stack_5 = torch.stack([x_96, x_99], dim=1)
        x_96 = x_99 = None
        sum_11 = stack_5.sum(1)
        x_100 = sum_11.mean((2, 3), keepdim=True)
        sum_11 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_layer2_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_105 = x_104.view(1, 2, 128, 1, 1)
        x_104 = None
        x_106 = torch.softmax(x_105, dim=1)
        x_105 = None
        mul_5 = stack_5 * x_106
        stack_5 = x_106 = None
        x_107 = torch.sum(mul_5, dim=1)
        mul_5 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_layer2_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_2_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_109 += x_93
        x_110 = x_109
        x_109 = x_93 = None
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        split_6 = torch.functional.split(x_111, 64, 1)
        getitem_42 = split_6[0]
        getitem_43 = split_6[1]
        split_6 = None
        x_112 = torch.conv2d(
            getitem_42,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_42 = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            getitem_43,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_43 = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        stack_6 = torch.stack([x_114, x_117], dim=1)
        x_114 = x_117 = None
        sum_13 = stack_6.sum(1)
        x_118 = sum_13.mean((2, 3), keepdim=True)
        sum_13 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_layer2_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_123 = x_122.view(1, 2, 128, 1, 1)
        x_122 = None
        x_124 = torch.softmax(x_123, dim=1)
        x_123 = None
        mul_6 = stack_6 * x_124
        stack_6 = x_124 = None
        x_125 = torch.sum(mul_6, dim=1)
        mul_6 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_layer2_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_3_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_127 += x_111
        x_128 = x_127
        x_127 = x_111 = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        split_7 = torch.functional.split(x_129, 64, 1)
        getitem_49 = split_7[0]
        getitem_50 = split_7[1]
        split_7 = None
        x_130 = torch.conv2d(
            getitem_49,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_49 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            getitem_50,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_50 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        stack_7 = torch.stack([x_132, x_135], dim=1)
        x_132 = x_135 = None
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        sum_15 = stack_7.sum(1)
        x_136 = sum_15.mean((2, 3), keepdim=True)
        sum_15 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_141 = x_140.view(1, 2, 256, 1, 1)
        x_140 = None
        x_142 = torch.softmax(x_141, dim=1)
        x_141 = None
        mul_7 = stack_7 * x_142
        stack_7 = x_142 = None
        x_143 = torch.sum(mul_7, dim=1)
        mul_7 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_3 = torch.conv2d(
            x_129,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_129 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_4 = torch.nn.functional.batch_norm(
            input_3,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_3 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_145 += input_4
        x_146 = x_145
        x_145 = input_4 = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        split_8 = torch.functional.split(x_147, 128, 1)
        getitem_56 = split_8[0]
        getitem_57 = split_8[1]
        split_8 = None
        x_148 = torch.conv2d(
            getitem_56,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_56 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_149 = torch.nn.functional.batch_norm(
            x_148,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_148 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_150 = torch.nn.functional.relu(x_149, inplace=True)
        x_149 = None
        x_151 = torch.conv2d(
            getitem_57,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_57 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_152 = torch.nn.functional.batch_norm(
            x_151,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_151 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_153 = torch.nn.functional.relu(x_152, inplace=True)
        x_152 = None
        stack_8 = torch.stack([x_150, x_153], dim=1)
        x_150 = x_153 = None
        sum_17 = stack_8.sum(1)
        x_154 = sum_17.mean((2, 3), keepdim=True)
        sum_17 = None
        x_155 = torch.conv2d(
            x_154,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_154 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_156 = torch.nn.functional.batch_norm(
            x_155,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_155 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_157 = torch.nn.functional.relu(x_156, inplace=True)
        x_156 = None
        x_158 = torch.conv2d(
            x_157,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_157 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_159 = x_158.view(1, 2, 256, 1, 1)
        x_158 = None
        x_160 = torch.softmax(x_159, dim=1)
        x_159 = None
        mul_8 = stack_8 * x_160
        stack_8 = x_160 = None
        x_161 = torch.sum(mul_8, dim=1)
        mul_8 = None
        x_162 = torch.conv2d(
            x_161,
            l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_161 = l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_163 = torch.nn.functional.batch_norm(
            x_162,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_162 = l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_163 += x_147
        x_164 = x_163
        x_163 = x_147 = None
        x_165 = torch.nn.functional.relu(x_164, inplace=True)
        x_164 = None
        split_9 = torch.functional.split(x_165, 128, 1)
        getitem_63 = split_9[0]
        getitem_64 = split_9[1]
        split_9 = None
        x_166 = torch.conv2d(
            getitem_63,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_63 = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_167 = torch.nn.functional.batch_norm(
            x_166,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_166 = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_168 = torch.nn.functional.relu(x_167, inplace=True)
        x_167 = None
        x_169 = torch.conv2d(
            getitem_64,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_64 = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_170 = torch.nn.functional.batch_norm(
            x_169,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_169 = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_171 = torch.nn.functional.relu(x_170, inplace=True)
        x_170 = None
        stack_9 = torch.stack([x_168, x_171], dim=1)
        x_168 = x_171 = None
        sum_19 = stack_9.sum(1)
        x_172 = sum_19.mean((2, 3), keepdim=True)
        sum_19 = None
        x_173 = torch.conv2d(
            x_172,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_172 = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_174 = torch.nn.functional.batch_norm(
            x_173,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_173 = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_175 = torch.nn.functional.relu(x_174, inplace=True)
        x_174 = None
        x_176 = torch.conv2d(
            x_175,
            l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_175 = l_self_modules_layer3_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_177 = x_176.view(1, 2, 256, 1, 1)
        x_176 = None
        x_178 = torch.softmax(x_177, dim=1)
        x_177 = None
        mul_9 = stack_9 * x_178
        stack_9 = x_178 = None
        x_179 = torch.sum(mul_9, dim=1)
        mul_9 = None
        x_180 = torch.conv2d(
            x_179,
            l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_179 = l_self_modules_layer3_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_181 = torch.nn.functional.batch_norm(
            x_180,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_180 = l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_2_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_181 += x_165
        x_182 = x_181
        x_181 = x_165 = None
        x_183 = torch.nn.functional.relu(x_182, inplace=True)
        x_182 = None
        split_10 = torch.functional.split(x_183, 128, 1)
        getitem_70 = split_10[0]
        getitem_71 = split_10[1]
        split_10 = None
        x_184 = torch.conv2d(
            getitem_70,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_70 = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_185 = torch.nn.functional.batch_norm(
            x_184,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_184 = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_186 = torch.nn.functional.relu(x_185, inplace=True)
        x_185 = None
        x_187 = torch.conv2d(
            getitem_71,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_71 = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_188 = torch.nn.functional.batch_norm(
            x_187,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_187 = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_189 = torch.nn.functional.relu(x_188, inplace=True)
        x_188 = None
        stack_10 = torch.stack([x_186, x_189], dim=1)
        x_186 = x_189 = None
        sum_21 = stack_10.sum(1)
        x_190 = sum_21.mean((2, 3), keepdim=True)
        sum_21 = None
        x_191 = torch.conv2d(
            x_190,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_190 = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_192 = torch.nn.functional.batch_norm(
            x_191,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_191 = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_193 = torch.nn.functional.relu(x_192, inplace=True)
        x_192 = None
        x_194 = torch.conv2d(
            x_193,
            l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_193 = l_self_modules_layer3_modules_3_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_195 = x_194.view(1, 2, 256, 1, 1)
        x_194 = None
        x_196 = torch.softmax(x_195, dim=1)
        x_195 = None
        mul_10 = stack_10 * x_196
        stack_10 = x_196 = None
        x_197 = torch.sum(mul_10, dim=1)
        mul_10 = None
        x_198 = torch.conv2d(
            x_197,
            l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_197 = l_self_modules_layer3_modules_3_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_199 = torch.nn.functional.batch_norm(
            x_198,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_198 = l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_3_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_199 += x_183
        x_200 = x_199
        x_199 = x_183 = None
        x_201 = torch.nn.functional.relu(x_200, inplace=True)
        x_200 = None
        split_11 = torch.functional.split(x_201, 128, 1)
        getitem_77 = split_11[0]
        getitem_78 = split_11[1]
        split_11 = None
        x_202 = torch.conv2d(
            getitem_77,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_77 = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_203 = torch.nn.functional.batch_norm(
            x_202,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_202 = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_204 = torch.nn.functional.relu(x_203, inplace=True)
        x_203 = None
        x_205 = torch.conv2d(
            getitem_78,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_78 = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_206 = torch.nn.functional.batch_norm(
            x_205,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_205 = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_207 = torch.nn.functional.relu(x_206, inplace=True)
        x_206 = None
        stack_11 = torch.stack([x_204, x_207], dim=1)
        x_204 = x_207 = None
        sum_23 = stack_11.sum(1)
        x_208 = sum_23.mean((2, 3), keepdim=True)
        sum_23 = None
        x_209 = torch.conv2d(
            x_208,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_208 = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_210 = torch.nn.functional.batch_norm(
            x_209,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_209 = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_211 = torch.nn.functional.relu(x_210, inplace=True)
        x_210 = None
        x_212 = torch.conv2d(
            x_211,
            l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_211 = l_self_modules_layer3_modules_4_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_213 = x_212.view(1, 2, 256, 1, 1)
        x_212 = None
        x_214 = torch.softmax(x_213, dim=1)
        x_213 = None
        mul_11 = stack_11 * x_214
        stack_11 = x_214 = None
        x_215 = torch.sum(mul_11, dim=1)
        mul_11 = None
        x_216 = torch.conv2d(
            x_215,
            l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_215 = l_self_modules_layer3_modules_4_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_217 = torch.nn.functional.batch_norm(
            x_216,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_216 = l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_4_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_217 += x_201
        x_218 = x_217
        x_217 = x_201 = None
        x_219 = torch.nn.functional.relu(x_218, inplace=True)
        x_218 = None
        split_12 = torch.functional.split(x_219, 128, 1)
        getitem_84 = split_12[0]
        getitem_85 = split_12[1]
        split_12 = None
        x_220 = torch.conv2d(
            getitem_84,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_84 = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_221 = torch.nn.functional.batch_norm(
            x_220,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_220 = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_222 = torch.nn.functional.relu(x_221, inplace=True)
        x_221 = None
        x_223 = torch.conv2d(
            getitem_85,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_85 = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_224 = torch.nn.functional.batch_norm(
            x_223,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_223 = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_225 = torch.nn.functional.relu(x_224, inplace=True)
        x_224 = None
        stack_12 = torch.stack([x_222, x_225], dim=1)
        x_222 = x_225 = None
        sum_25 = stack_12.sum(1)
        x_226 = sum_25.mean((2, 3), keepdim=True)
        sum_25 = None
        x_227 = torch.conv2d(
            x_226,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_226 = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_228 = torch.nn.functional.batch_norm(
            x_227,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_227 = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_229 = torch.nn.functional.relu(x_228, inplace=True)
        x_228 = None
        x_230 = torch.conv2d(
            x_229,
            l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_229 = l_self_modules_layer3_modules_5_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_231 = x_230.view(1, 2, 256, 1, 1)
        x_230 = None
        x_232 = torch.softmax(x_231, dim=1)
        x_231 = None
        mul_12 = stack_12 * x_232
        stack_12 = x_232 = None
        x_233 = torch.sum(mul_12, dim=1)
        mul_12 = None
        x_234 = torch.conv2d(
            x_233,
            l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_233 = l_self_modules_layer3_modules_5_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_235 = torch.nn.functional.batch_norm(
            x_234,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_234 = l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_5_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_235 += x_219
        x_236 = x_235
        x_235 = x_219 = None
        x_237 = torch.nn.functional.relu(x_236, inplace=True)
        x_236 = None
        split_13 = torch.functional.split(x_237, 128, 1)
        getitem_91 = split_13[0]
        getitem_92 = split_13[1]
        split_13 = None
        x_238 = torch.conv2d(
            getitem_91,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_91 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_239 = torch.nn.functional.batch_norm(
            x_238,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_238 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_240 = torch.nn.functional.relu(x_239, inplace=True)
        x_239 = None
        x_241 = torch.conv2d(
            getitem_92,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_92 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_242 = torch.nn.functional.batch_norm(
            x_241,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_241 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_243 = torch.nn.functional.relu(x_242, inplace=True)
        x_242 = None
        stack_13 = torch.stack([x_240, x_243], dim=1)
        x_240 = x_243 = None
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        sum_27 = stack_13.sum(1)
        x_244 = sum_27.mean((2, 3), keepdim=True)
        sum_27 = None
        x_245 = torch.conv2d(
            x_244,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_244 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_246 = torch.nn.functional.batch_norm(
            x_245,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_245 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_247 = torch.nn.functional.relu(x_246, inplace=True)
        x_246 = None
        x_248 = torch.conv2d(
            x_247,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_247 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_249 = x_248.view(1, 2, 512, 1, 1)
        x_248 = None
        x_250 = torch.softmax(x_249, dim=1)
        x_249 = None
        mul_13 = stack_13 * x_250
        stack_13 = x_250 = None
        x_251 = torch.sum(mul_13, dim=1)
        mul_13 = None
        x_252 = torch.conv2d(
            x_251,
            l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_251 = l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_253 = torch.nn.functional.batch_norm(
            x_252,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_252 = l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_5 = torch.conv2d(
            x_237,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_237 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
        input_6 = torch.nn.functional.batch_norm(
            input_5,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_5 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = (None)
        x_253 += input_6
        x_254 = x_253
        x_253 = input_6 = None
        x_255 = torch.nn.functional.relu(x_254, inplace=True)
        x_254 = None
        split_14 = torch.functional.split(x_255, 256, 1)
        getitem_98 = split_14[0]
        getitem_99 = split_14[1]
        split_14 = None
        x_256 = torch.conv2d(
            getitem_98,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_98 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_257 = torch.nn.functional.batch_norm(
            x_256,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_256 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_258 = torch.nn.functional.relu(x_257, inplace=True)
        x_257 = None
        x_259 = torch.conv2d(
            getitem_99,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_99 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_260 = torch.nn.functional.batch_norm(
            x_259,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_259 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_261 = torch.nn.functional.relu(x_260, inplace=True)
        x_260 = None
        stack_14 = torch.stack([x_258, x_261], dim=1)
        x_258 = x_261 = None
        sum_29 = stack_14.sum(1)
        x_262 = sum_29.mean((2, 3), keepdim=True)
        sum_29 = None
        x_263 = torch.conv2d(
            x_262,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_262 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_264 = torch.nn.functional.batch_norm(
            x_263,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_263 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_265 = torch.nn.functional.relu(x_264, inplace=True)
        x_264 = None
        x_266 = torch.conv2d(
            x_265,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_265 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_267 = x_266.view(1, 2, 512, 1, 1)
        x_266 = None
        x_268 = torch.softmax(x_267, dim=1)
        x_267 = None
        mul_14 = stack_14 * x_268
        stack_14 = x_268 = None
        x_269 = torch.sum(mul_14, dim=1)
        mul_14 = None
        x_270 = torch.conv2d(
            x_269,
            l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_269 = l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_271 = torch.nn.functional.batch_norm(
            x_270,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_270 = l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_271 += x_255
        x_272 = x_271
        x_271 = x_255 = None
        x_273 = torch.nn.functional.relu(x_272, inplace=True)
        x_272 = None
        split_15 = torch.functional.split(x_273, 256, 1)
        getitem_105 = split_15[0]
        getitem_106 = split_15[1]
        split_15 = None
        x_274 = torch.conv2d(
            getitem_105,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_105 = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_275 = torch.nn.functional.batch_norm(
            x_274,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_274 = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_276 = torch.nn.functional.relu(x_275, inplace=True)
        x_275 = None
        x_277 = torch.conv2d(
            getitem_106,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_106 = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_278 = torch.nn.functional.batch_norm(
            x_277,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_277 = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_279 = torch.nn.functional.relu(x_278, inplace=True)
        x_278 = None
        stack_15 = torch.stack([x_276, x_279], dim=1)
        x_276 = x_279 = None
        sum_31 = stack_15.sum(1)
        x_280 = sum_31.mean((2, 3), keepdim=True)
        sum_31 = None
        x_281 = torch.conv2d(
            x_280,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_280 = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_282 = torch.nn.functional.batch_norm(
            x_281,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_281 = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_283 = torch.nn.functional.relu(x_282, inplace=True)
        x_282 = None
        x_284 = torch.conv2d(
            x_283,
            l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_283 = l_self_modules_layer4_modules_2_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_285 = x_284.view(1, 2, 512, 1, 1)
        x_284 = None
        x_286 = torch.softmax(x_285, dim=1)
        x_285 = None
        mul_15 = stack_15 * x_286
        stack_15 = x_286 = None
        x_287 = torch.sum(mul_15, dim=1)
        mul_15 = None
        x_288 = torch.conv2d(
            x_287,
            l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_287 = l_self_modules_layer4_modules_2_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_289 = torch.nn.functional.batch_norm(
            x_288,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_288 = l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_2_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_289 += x_273
        x_290 = x_289
        x_289 = x_273 = None
        x_291 = torch.nn.functional.relu(x_290, inplace=True)
        x_290 = None
        x_292 = torch.nn.functional.adaptive_avg_pool2d(x_291, 1)
        x_291 = None
        x_293 = x_292.flatten(1, -1)
        x_292 = None
        x_294 = torch._C._nn.linear(
            x_293,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_293 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_294,)
