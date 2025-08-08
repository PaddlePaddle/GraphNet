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
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_14 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_41 = torch.nn.functional.batch_norm(
            x_40,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_40 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_42 = torch.nn.functional.relu(x_41, inplace=True)
        x_41 = None
        x_43 = torch.conv2d(
            getitem_15,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_15 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_44 = torch.nn.functional.batch_norm(
            x_43,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_43 = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_45 = torch.nn.functional.relu(x_44, inplace=True)
        x_44 = None
        stack_2 = torch.stack([x_42, x_45], dim=1)
        x_42 = x_45 = None
        floordiv_1 = sym_sum // 8
        sym_sum_2 = torch.sym_sum([1, floordiv_1])
        floordiv_1 = sym_sum_2 = None
        sum_5 = stack_2.sum(1)
        x_46 = sum_5.mean((2, 3), keepdim=True)
        sum_5 = None
        x_47 = torch.conv2d(
            x_46,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_46 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_48 = torch.nn.functional.batch_norm(
            x_47,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_47 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_49 = torch.nn.functional.relu(x_48, inplace=True)
        x_48 = None
        x_50 = torch.conv2d(
            x_49,
            l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_49 = l_self_modules_layer2_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_51 = x_50.view(1, 2, 128, 1, 1)
        x_50 = None
        x_52 = torch.softmax(x_51, dim=1)
        x_51 = None
        mul_2 = stack_2 * x_52
        stack_2 = x_52 = None
        x_53 = torch.sum(mul_2, dim=1)
        mul_2 = None
        x_54 = torch.conv2d(
            x_53,
            l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_53 = l_self_modules_layer2_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_55 = torch.nn.functional.batch_norm(
            x_54,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_54 = l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_1 = torch.conv2d(
            x_39,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_39 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        x_55 += input_2
        x_56 = x_55
        x_55 = input_2 = None
        x_57 = torch.nn.functional.relu(x_56, inplace=True)
        x_56 = None
        split_3 = torch.functional.split(x_57, 64, 1)
        getitem_21 = split_3[0]
        getitem_22 = split_3[1]
        split_3 = None
        x_58 = torch.conv2d(
            getitem_21,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_21 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_59 = torch.nn.functional.batch_norm(
            x_58,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_58 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_60 = torch.nn.functional.relu(x_59, inplace=True)
        x_59 = None
        x_61 = torch.conv2d(
            getitem_22,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_22 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_62 = torch.nn.functional.batch_norm(
            x_61,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_61 = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_63 = torch.nn.functional.relu(x_62, inplace=True)
        x_62 = None
        stack_3 = torch.stack([x_60, x_63], dim=1)
        x_60 = x_63 = None
        sum_7 = stack_3.sum(1)
        x_64 = sum_7.mean((2, 3), keepdim=True)
        sum_7 = None
        x_65 = torch.conv2d(
            x_64,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_64 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_66 = torch.nn.functional.batch_norm(
            x_65,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_65 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_67 = torch.nn.functional.relu(x_66, inplace=True)
        x_66 = None
        x_68 = torch.conv2d(
            x_67,
            l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_67 = l_self_modules_layer2_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
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
            l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_71 = l_self_modules_layer2_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_73 = torch.nn.functional.batch_norm(
            x_72,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_72 = l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer2_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_73 += x_57
        x_74 = x_73
        x_73 = x_57 = None
        x_75 = torch.nn.functional.relu(x_74, inplace=True)
        x_74 = None
        split_4 = torch.functional.split(x_75, 64, 1)
        getitem_28 = split_4[0]
        getitem_29 = split_4[1]
        split_4 = None
        x_76 = torch.conv2d(
            getitem_28,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_28 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_77 = torch.nn.functional.batch_norm(
            x_76,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_76 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_78 = torch.nn.functional.relu(x_77, inplace=True)
        x_77 = None
        x_79 = torch.conv2d(
            getitem_29,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_29 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_80 = torch.nn.functional.batch_norm(
            x_79,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_79 = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_81 = torch.nn.functional.relu(x_80, inplace=True)
        x_80 = None
        stack_4 = torch.stack([x_78, x_81], dim=1)
        x_78 = x_81 = None
        floordiv_2 = sym_sum // 16
        sym_sum_3 = torch.sym_sum([1, floordiv_2])
        floordiv_2 = sym_sum_3 = None
        sum_9 = stack_4.sum(1)
        x_82 = sum_9.mean((2, 3), keepdim=True)
        sum_9 = None
        x_83 = torch.conv2d(
            x_82,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_82 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_84 = torch.nn.functional.batch_norm(
            x_83,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_83 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_85 = torch.nn.functional.relu(x_84, inplace=True)
        x_84 = None
        x_86 = torch.conv2d(
            x_85,
            l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_85 = l_self_modules_layer3_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_87 = x_86.view(1, 2, 256, 1, 1)
        x_86 = None
        x_88 = torch.softmax(x_87, dim=1)
        x_87 = None
        mul_4 = stack_4 * x_88
        stack_4 = x_88 = None
        x_89 = torch.sum(mul_4, dim=1)
        mul_4 = None
        x_90 = torch.conv2d(
            x_89,
            l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_89 = l_self_modules_layer3_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_91 = torch.nn.functional.batch_norm(
            x_90,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_90 = l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_3 = torch.conv2d(
            x_75,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_75 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        x_91 += input_4
        x_92 = x_91
        x_91 = input_4 = None
        x_93 = torch.nn.functional.relu(x_92, inplace=True)
        x_92 = None
        split_5 = torch.functional.split(x_93, 128, 1)
        getitem_35 = split_5[0]
        getitem_36 = split_5[1]
        split_5 = None
        x_94 = torch.conv2d(
            getitem_35,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_35 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_95 = torch.nn.functional.batch_norm(
            x_94,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_94 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_96 = torch.nn.functional.relu(x_95, inplace=True)
        x_95 = None
        x_97 = torch.conv2d(
            getitem_36,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_36 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_98 = torch.nn.functional.batch_norm(
            x_97,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_97 = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_99 = torch.nn.functional.relu(x_98, inplace=True)
        x_98 = None
        stack_5 = torch.stack([x_96, x_99], dim=1)
        x_96 = x_99 = None
        sum_11 = stack_5.sum(1)
        x_100 = sum_11.mean((2, 3), keepdim=True)
        sum_11 = None
        x_101 = torch.conv2d(
            x_100,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_100 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_102 = torch.nn.functional.batch_norm(
            x_101,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_101 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_103 = torch.nn.functional.relu(x_102, inplace=True)
        x_102 = None
        x_104 = torch.conv2d(
            x_103,
            l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_103 = l_self_modules_layer3_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_105 = x_104.view(1, 2, 256, 1, 1)
        x_104 = None
        x_106 = torch.softmax(x_105, dim=1)
        x_105 = None
        mul_5 = stack_5 * x_106
        stack_5 = x_106 = None
        x_107 = torch.sum(mul_5, dim=1)
        mul_5 = None
        x_108 = torch.conv2d(
            x_107,
            l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_107 = l_self_modules_layer3_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_109 = torch.nn.functional.batch_norm(
            x_108,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_108 = l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer3_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_109 += x_93
        x_110 = x_109
        x_109 = x_93 = None
        x_111 = torch.nn.functional.relu(x_110, inplace=True)
        x_110 = None
        split_6 = torch.functional.split(x_111, 128, 1)
        getitem_42 = split_6[0]
        getitem_43 = split_6[1]
        split_6 = None
        x_112 = torch.conv2d(
            getitem_42,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_42 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_113 = torch.nn.functional.batch_norm(
            x_112,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_112 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_114 = torch.nn.functional.relu(x_113, inplace=True)
        x_113 = None
        x_115 = torch.conv2d(
            getitem_43,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (2, 2),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_43 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_116 = torch.nn.functional.batch_norm(
            x_115,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_115 = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_117 = torch.nn.functional.relu(x_116, inplace=True)
        x_116 = None
        stack_6 = torch.stack([x_114, x_117], dim=1)
        x_114 = x_117 = None
        floordiv_3 = sym_sum // 32
        sym_sum = None
        sym_sum_4 = torch.sym_sum([1, floordiv_3])
        floordiv_3 = sym_sum_4 = None
        sum_13 = stack_6.sum(1)
        x_118 = sum_13.mean((2, 3), keepdim=True)
        sum_13 = None
        x_119 = torch.conv2d(
            x_118,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_118 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_120 = torch.nn.functional.batch_norm(
            x_119,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_119 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_121 = torch.nn.functional.relu(x_120, inplace=True)
        x_120 = None
        x_122 = torch.conv2d(
            x_121,
            l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_121 = l_self_modules_layer4_modules_0_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_123 = x_122.view(1, 2, 512, 1, 1)
        x_122 = None
        x_124 = torch.softmax(x_123, dim=1)
        x_123 = None
        mul_6 = stack_6 * x_124
        stack_6 = x_124 = None
        x_125 = torch.sum(mul_6, dim=1)
        mul_6 = None
        x_126 = torch.conv2d(
            x_125,
            l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_125 = l_self_modules_layer4_modules_0_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_127 = torch.nn.functional.batch_norm(
            x_126,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_126 = l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_conv2_modules_bn_parameters_bias_
        ) = None
        input_5 = torch.conv2d(
            x_111,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        x_111 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        x_127 += input_6
        x_128 = x_127
        x_127 = input_6 = None
        x_129 = torch.nn.functional.relu(x_128, inplace=True)
        x_128 = None
        split_7 = torch.functional.split(x_129, 256, 1)
        getitem_49 = split_7[0]
        getitem_50 = split_7[1]
        split_7 = None
        x_130 = torch.conv2d(
            getitem_49,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        getitem_49 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_conv_parameters_weight_ = (None)
        x_131 = torch.nn.functional.batch_norm(
            x_130,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_130 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_0_modules_bn_parameters_bias_ = (None)
        x_132 = torch.nn.functional.relu(x_131, inplace=True)
        x_131 = None
        x_133 = torch.conv2d(
            getitem_50,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (2, 2),
            (2, 2),
            1,
        )
        getitem_50 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_conv_parameters_weight_ = (None)
        x_134 = torch.nn.functional.batch_norm(
            x_133,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_133 = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_paths_modules_1_modules_bn_parameters_bias_ = (None)
        x_135 = torch.nn.functional.relu(x_134, inplace=True)
        x_134 = None
        stack_7 = torch.stack([x_132, x_135], dim=1)
        x_132 = x_135 = None
        sum_15 = stack_7.sum(1)
        x_136 = sum_15.mean((2, 3), keepdim=True)
        sum_15 = None
        x_137 = torch.conv2d(
            x_136,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_136 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_reduce_parameters_weight_ = (None)
        x_138 = torch.nn.functional.batch_norm(
            x_137,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_137 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_buffers_running_var_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_weight_ = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_bn_parameters_bias_ = (None)
        x_139 = torch.nn.functional.relu(x_138, inplace=True)
        x_138 = None
        x_140 = torch.conv2d(
            x_139,
            l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_139 = l_self_modules_layer4_modules_1_modules_conv1_modules_attn_modules_fc_select_parameters_weight_ = (None)
        x_141 = x_140.view(1, 2, 512, 1, 1)
        x_140 = None
        x_142 = torch.softmax(x_141, dim=1)
        x_141 = None
        mul_7 = stack_7 * x_142
        stack_7 = x_142 = None
        x_143 = torch.sum(mul_7, dim=1)
        mul_7 = None
        x_144 = torch.conv2d(
            x_143,
            l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        x_143 = l_self_modules_layer4_modules_1_modules_conv2_modules_conv_parameters_weight_ = (None)
        x_145 = torch.nn.functional.batch_norm(
            x_144,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        x_144 = l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_mean_ = l_self_modules_layer4_modules_1_modules_conv2_modules_bn_buffers_running_var_ = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_conv2_modules_bn_parameters_bias_
        ) = None
        x_145 += x_129
        x_146 = x_145
        x_145 = x_129 = None
        x_147 = torch.nn.functional.relu(x_146, inplace=True)
        x_146 = None
        x_148 = torch.nn.functional.adaptive_avg_pool2d(x_147, 1)
        x_147 = None
        x_149 = x_148.flatten(1, -1)
        x_148 = None
        x_150 = torch._C._nn.linear(
            x_149,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_149 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_150,)
