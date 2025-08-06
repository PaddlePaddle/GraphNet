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
        L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_bias_
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bn3_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bn1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_conv3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bn3_parameters_bias_
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
        out = torch.conv2d(
            x_3,
            l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_0_modules_conv1_parameters_weight_ = None
        out_1 = torch.nn.functional.batch_norm(
            out,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn1_parameters_bias_ = None
        out_2 = torch.nn.functional.relu(out_1, inplace=True)
        out_1 = None
        split = torch.functional.split(out_2, 32, 1)
        out_2 = None
        sp = split[0]
        sp_4 = split[1]
        sp_8 = split[2]
        getitem_3 = split[3]
        split = None
        sp_1 = torch.conv2d(
            sp,
            l_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp = (
            l_self_modules_layer1_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_2 = torch.nn.functional.batch_norm(
            sp_1,
            l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_1 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_3 = torch.nn.functional.relu(sp_2, inplace=True)
        sp_2 = None
        sp_5 = torch.conv2d(
            sp_4,
            l_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_4 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_6 = torch.nn.functional.batch_norm(
            sp_5,
            l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_5 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_7 = torch.nn.functional.relu(sp_6, inplace=True)
        sp_6 = None
        sp_9 = torch.conv2d(
            sp_8,
            l_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_8 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_10 = torch.nn.functional.batch_norm(
            sp_9,
            l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_9 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_11 = torch.nn.functional.relu(sp_10, inplace=True)
        sp_10 = None
        avg_pool2d = torch._C._nn.avg_pool2d(getitem_3, 3, 1, 1, False, True, None)
        getitem_3 = None
        out_3 = torch.cat([sp_3, sp_7, sp_11, avg_pool2d], 1)
        sp_3 = sp_7 = sp_11 = avg_pool2d = None
        out_4 = torch.conv2d(
            out_3,
            l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_3 = l_self_modules_layer1_modules_0_modules_conv3_parameters_weight_ = None
        out_5 = torch.nn.functional.batch_norm(
            out_4,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_4 = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_0_modules_bn3_parameters_bias_ = None
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
        out_5 += input_2
        out_6 = out_5
        out_5 = input_2 = None
        out_7 = torch.nn.functional.relu(out_6, inplace=True)
        out_6 = None
        out_8 = torch.conv2d(
            out_7,
            l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_1_modules_conv1_parameters_weight_ = None
        out_9 = torch.nn.functional.batch_norm(
            out_8,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_8 = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn1_parameters_bias_ = None
        out_10 = torch.nn.functional.relu(out_9, inplace=True)
        out_9 = None
        split_1 = torch.functional.split(out_10, 32, 1)
        out_10 = None
        sp_12 = split_1[0]
        getitem_5 = split_1[1]
        getitem_6 = split_1[2]
        getitem_7 = split_1[3]
        split_1 = None
        sp_13 = torch.conv2d(
            sp_12,
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_12 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_14 = torch.nn.functional.batch_norm(
            sp_13,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_13 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_15 = torch.nn.functional.relu(sp_14, inplace=True)
        sp_14 = None
        sp_16 = sp_15 + getitem_5
        getitem_5 = None
        sp_17 = torch.conv2d(
            sp_16,
            l_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_16 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_18 = torch.nn.functional.batch_norm(
            sp_17,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_17 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_19 = torch.nn.functional.relu(sp_18, inplace=True)
        sp_18 = None
        sp_20 = sp_19 + getitem_6
        getitem_6 = None
        sp_21 = torch.conv2d(
            sp_20,
            l_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_20 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_22 = torch.nn.functional.batch_norm(
            sp_21,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_21 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_23 = torch.nn.functional.relu(sp_22, inplace=True)
        sp_22 = None
        out_11 = torch.cat([sp_15, sp_19, sp_23, getitem_7], 1)
        sp_15 = sp_19 = sp_23 = getitem_7 = None
        out_12 = torch.conv2d(
            out_11,
            l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_11 = l_self_modules_layer1_modules_1_modules_conv3_parameters_weight_ = None
        out_13 = torch.nn.functional.batch_norm(
            out_12,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_12 = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_1_modules_bn3_parameters_bias_ = None
        out_13 += out_7
        out_14 = out_13
        out_13 = out_7 = None
        out_15 = torch.nn.functional.relu(out_14, inplace=True)
        out_14 = None
        out_16 = torch.conv2d(
            out_15,
            l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer1_modules_2_modules_conv1_parameters_weight_ = None
        out_17 = torch.nn.functional.batch_norm(
            out_16,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_16 = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn1_parameters_bias_ = None
        out_18 = torch.nn.functional.relu(out_17, inplace=True)
        out_17 = None
        split_2 = torch.functional.split(out_18, 32, 1)
        out_18 = None
        sp_24 = split_2[0]
        getitem_9 = split_2[1]
        getitem_10 = split_2[2]
        getitem_11 = split_2[3]
        split_2 = None
        sp_25 = torch.conv2d(
            sp_24,
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_24 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_26 = torch.nn.functional.batch_norm(
            sp_25,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_25 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_27 = torch.nn.functional.relu(sp_26, inplace=True)
        sp_26 = None
        sp_28 = sp_27 + getitem_9
        getitem_9 = None
        sp_29 = torch.conv2d(
            sp_28,
            l_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_28 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_30 = torch.nn.functional.batch_norm(
            sp_29,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_29 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_31 = torch.nn.functional.relu(sp_30, inplace=True)
        sp_30 = None
        sp_32 = sp_31 + getitem_10
        getitem_10 = None
        sp_33 = torch.conv2d(
            sp_32,
            l_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_32 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_34 = torch.nn.functional.batch_norm(
            sp_33,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_33 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_35 = torch.nn.functional.relu(sp_34, inplace=True)
        sp_34 = None
        out_19 = torch.cat([sp_27, sp_31, sp_35, getitem_11], 1)
        sp_27 = sp_31 = sp_35 = getitem_11 = None
        out_20 = torch.conv2d(
            out_19,
            l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_19 = l_self_modules_layer1_modules_2_modules_conv3_parameters_weight_ = None
        out_21 = torch.nn.functional.batch_norm(
            out_20,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_20 = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer1_modules_2_modules_bn3_parameters_bias_ = None
        out_21 += out_15
        out_22 = out_21
        out_21 = out_15 = None
        out_23 = torch.nn.functional.relu(out_22, inplace=True)
        out_22 = None
        out_24 = torch.conv2d(
            out_23,
            l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_0_modules_conv1_parameters_weight_ = None
        out_25 = torch.nn.functional.batch_norm(
            out_24,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_24 = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn1_parameters_bias_ = None
        out_26 = torch.nn.functional.relu(out_25, inplace=True)
        out_25 = None
        split_3 = torch.functional.split(out_26, 64, 1)
        out_26 = None
        sp_36 = split_3[0]
        sp_40 = split_3[1]
        sp_44 = split_3[2]
        getitem_15 = split_3[3]
        split_3 = None
        sp_37 = torch.conv2d(
            sp_36,
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_36 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_38 = torch.nn.functional.batch_norm(
            sp_37,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_37 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_39 = torch.nn.functional.relu(sp_38, inplace=True)
        sp_38 = None
        sp_41 = torch.conv2d(
            sp_40,
            l_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_40 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_42 = torch.nn.functional.batch_norm(
            sp_41,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_41 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_43 = torch.nn.functional.relu(sp_42, inplace=True)
        sp_42 = None
        sp_45 = torch.conv2d(
            sp_44,
            l_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_44 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_46 = torch.nn.functional.batch_norm(
            sp_45,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_45 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_47 = torch.nn.functional.relu(sp_46, inplace=True)
        sp_46 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(getitem_15, 3, 2, 1, False, True, None)
        getitem_15 = None
        out_27 = torch.cat([sp_39, sp_43, sp_47, avg_pool2d_1], 1)
        sp_39 = sp_43 = sp_47 = avg_pool2d_1 = None
        out_28 = torch.conv2d(
            out_27,
            l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_27 = l_self_modules_layer2_modules_0_modules_conv3_parameters_weight_ = None
        out_29 = torch.nn.functional.batch_norm(
            out_28,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_28 = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_0_modules_bn3_parameters_bias_ = None
        input_3 = torch.conv2d(
            out_23,
            l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_23 = l_self_modules_layer2_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        out_29 += input_4
        out_30 = out_29
        out_29 = input_4 = None
        out_31 = torch.nn.functional.relu(out_30, inplace=True)
        out_30 = None
        out_32 = torch.conv2d(
            out_31,
            l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_1_modules_conv1_parameters_weight_ = None
        out_33 = torch.nn.functional.batch_norm(
            out_32,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_32 = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn1_parameters_bias_ = None
        out_34 = torch.nn.functional.relu(out_33, inplace=True)
        out_33 = None
        split_4 = torch.functional.split(out_34, 64, 1)
        out_34 = None
        sp_48 = split_4[0]
        getitem_17 = split_4[1]
        getitem_18 = split_4[2]
        getitem_19 = split_4[3]
        split_4 = None
        sp_49 = torch.conv2d(
            sp_48,
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_48 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_50 = torch.nn.functional.batch_norm(
            sp_49,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_49 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_51 = torch.nn.functional.relu(sp_50, inplace=True)
        sp_50 = None
        sp_52 = sp_51 + getitem_17
        getitem_17 = None
        sp_53 = torch.conv2d(
            sp_52,
            l_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_52 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_54 = torch.nn.functional.batch_norm(
            sp_53,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_53 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_55 = torch.nn.functional.relu(sp_54, inplace=True)
        sp_54 = None
        sp_56 = sp_55 + getitem_18
        getitem_18 = None
        sp_57 = torch.conv2d(
            sp_56,
            l_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_56 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_58 = torch.nn.functional.batch_norm(
            sp_57,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_57 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_59 = torch.nn.functional.relu(sp_58, inplace=True)
        sp_58 = None
        out_35 = torch.cat([sp_51, sp_55, sp_59, getitem_19], 1)
        sp_51 = sp_55 = sp_59 = getitem_19 = None
        out_36 = torch.conv2d(
            out_35,
            l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_35 = l_self_modules_layer2_modules_1_modules_conv3_parameters_weight_ = None
        out_37 = torch.nn.functional.batch_norm(
            out_36,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_36 = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_1_modules_bn3_parameters_bias_ = None
        out_37 += out_31
        out_38 = out_37
        out_37 = out_31 = None
        out_39 = torch.nn.functional.relu(out_38, inplace=True)
        out_38 = None
        out_40 = torch.conv2d(
            out_39,
            l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_2_modules_conv1_parameters_weight_ = None
        out_41 = torch.nn.functional.batch_norm(
            out_40,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_40 = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn1_parameters_bias_ = None
        out_42 = torch.nn.functional.relu(out_41, inplace=True)
        out_41 = None
        split_5 = torch.functional.split(out_42, 64, 1)
        out_42 = None
        sp_60 = split_5[0]
        getitem_21 = split_5[1]
        getitem_22 = split_5[2]
        getitem_23 = split_5[3]
        split_5 = None
        sp_61 = torch.conv2d(
            sp_60,
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_60 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_62 = torch.nn.functional.batch_norm(
            sp_61,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_61 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_63 = torch.nn.functional.relu(sp_62, inplace=True)
        sp_62 = None
        sp_64 = sp_63 + getitem_21
        getitem_21 = None
        sp_65 = torch.conv2d(
            sp_64,
            l_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_64 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_66 = torch.nn.functional.batch_norm(
            sp_65,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_65 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_67 = torch.nn.functional.relu(sp_66, inplace=True)
        sp_66 = None
        sp_68 = sp_67 + getitem_22
        getitem_22 = None
        sp_69 = torch.conv2d(
            sp_68,
            l_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_68 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_70 = torch.nn.functional.batch_norm(
            sp_69,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_69 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_71 = torch.nn.functional.relu(sp_70, inplace=True)
        sp_70 = None
        out_43 = torch.cat([sp_63, sp_67, sp_71, getitem_23], 1)
        sp_63 = sp_67 = sp_71 = getitem_23 = None
        out_44 = torch.conv2d(
            out_43,
            l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_43 = l_self_modules_layer2_modules_2_modules_conv3_parameters_weight_ = None
        out_45 = torch.nn.functional.batch_norm(
            out_44,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_44 = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_2_modules_bn3_parameters_bias_ = None
        out_45 += out_39
        out_46 = out_45
        out_45 = out_39 = None
        out_47 = torch.nn.functional.relu(out_46, inplace=True)
        out_46 = None
        out_48 = torch.conv2d(
            out_47,
            l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer2_modules_3_modules_conv1_parameters_weight_ = None
        out_49 = torch.nn.functional.batch_norm(
            out_48,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_48 = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn1_parameters_bias_ = None
        out_50 = torch.nn.functional.relu(out_49, inplace=True)
        out_49 = None
        split_6 = torch.functional.split(out_50, 64, 1)
        out_50 = None
        sp_72 = split_6[0]
        getitem_25 = split_6[1]
        getitem_26 = split_6[2]
        getitem_27 = split_6[3]
        split_6 = None
        sp_73 = torch.conv2d(
            sp_72,
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_72 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_74 = torch.nn.functional.batch_norm(
            sp_73,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_73 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_75 = torch.nn.functional.relu(sp_74, inplace=True)
        sp_74 = None
        sp_76 = sp_75 + getitem_25
        getitem_25 = None
        sp_77 = torch.conv2d(
            sp_76,
            l_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_76 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_78 = torch.nn.functional.batch_norm(
            sp_77,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_77 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_79 = torch.nn.functional.relu(sp_78, inplace=True)
        sp_78 = None
        sp_80 = sp_79 + getitem_26
        getitem_26 = None
        sp_81 = torch.conv2d(
            sp_80,
            l_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_80 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_82 = torch.nn.functional.batch_norm(
            sp_81,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_81 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_83 = torch.nn.functional.relu(sp_82, inplace=True)
        sp_82 = None
        out_51 = torch.cat([sp_75, sp_79, sp_83, getitem_27], 1)
        sp_75 = sp_79 = sp_83 = getitem_27 = None
        out_52 = torch.conv2d(
            out_51,
            l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_51 = l_self_modules_layer2_modules_3_modules_conv3_parameters_weight_ = None
        out_53 = torch.nn.functional.batch_norm(
            out_52,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_52 = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer2_modules_3_modules_bn3_parameters_bias_ = None
        out_53 += out_47
        out_54 = out_53
        out_53 = out_47 = None
        out_55 = torch.nn.functional.relu(out_54, inplace=True)
        out_54 = None
        out_56 = torch.conv2d(
            out_55,
            l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_0_modules_conv1_parameters_weight_ = None
        out_57 = torch.nn.functional.batch_norm(
            out_56,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_56 = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn1_parameters_bias_ = None
        out_58 = torch.nn.functional.relu(out_57, inplace=True)
        out_57 = None
        split_7 = torch.functional.split(out_58, 128, 1)
        out_58 = None
        sp_84 = split_7[0]
        sp_88 = split_7[1]
        sp_92 = split_7[2]
        getitem_31 = split_7[3]
        split_7 = None
        sp_85 = torch.conv2d(
            sp_84,
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_84 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_86 = torch.nn.functional.batch_norm(
            sp_85,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_85 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_87 = torch.nn.functional.relu(sp_86, inplace=True)
        sp_86 = None
        sp_89 = torch.conv2d(
            sp_88,
            l_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_88 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_90 = torch.nn.functional.batch_norm(
            sp_89,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_89 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_91 = torch.nn.functional.relu(sp_90, inplace=True)
        sp_90 = None
        sp_93 = torch.conv2d(
            sp_92,
            l_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_92 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_94 = torch.nn.functional.batch_norm(
            sp_93,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_93 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_95 = torch.nn.functional.relu(sp_94, inplace=True)
        sp_94 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(getitem_31, 3, 2, 1, False, True, None)
        getitem_31 = None
        out_59 = torch.cat([sp_87, sp_91, sp_95, avg_pool2d_2], 1)
        sp_87 = sp_91 = sp_95 = avg_pool2d_2 = None
        out_60 = torch.conv2d(
            out_59,
            l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_59 = l_self_modules_layer3_modules_0_modules_conv3_parameters_weight_ = None
        out_61 = torch.nn.functional.batch_norm(
            out_60,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_60 = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_0_modules_bn3_parameters_bias_ = None
        input_5 = torch.conv2d(
            out_55,
            l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_55 = l_self_modules_layer3_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        out_61 += input_6
        out_62 = out_61
        out_61 = input_6 = None
        out_63 = torch.nn.functional.relu(out_62, inplace=True)
        out_62 = None
        out_64 = torch.conv2d(
            out_63,
            l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_1_modules_conv1_parameters_weight_ = None
        out_65 = torch.nn.functional.batch_norm(
            out_64,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_64 = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn1_parameters_bias_ = None
        out_66 = torch.nn.functional.relu(out_65, inplace=True)
        out_65 = None
        split_8 = torch.functional.split(out_66, 128, 1)
        out_66 = None
        sp_96 = split_8[0]
        getitem_33 = split_8[1]
        getitem_34 = split_8[2]
        getitem_35 = split_8[3]
        split_8 = None
        sp_97 = torch.conv2d(
            sp_96,
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_96 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_98 = torch.nn.functional.batch_norm(
            sp_97,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_97 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_99 = torch.nn.functional.relu(sp_98, inplace=True)
        sp_98 = None
        sp_100 = sp_99 + getitem_33
        getitem_33 = None
        sp_101 = torch.conv2d(
            sp_100,
            l_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_100 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_102 = torch.nn.functional.batch_norm(
            sp_101,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_101 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_103 = torch.nn.functional.relu(sp_102, inplace=True)
        sp_102 = None
        sp_104 = sp_103 + getitem_34
        getitem_34 = None
        sp_105 = torch.conv2d(
            sp_104,
            l_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_104 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_106 = torch.nn.functional.batch_norm(
            sp_105,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_105 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_107 = torch.nn.functional.relu(sp_106, inplace=True)
        sp_106 = None
        out_67 = torch.cat([sp_99, sp_103, sp_107, getitem_35], 1)
        sp_99 = sp_103 = sp_107 = getitem_35 = None
        out_68 = torch.conv2d(
            out_67,
            l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_67 = l_self_modules_layer3_modules_1_modules_conv3_parameters_weight_ = None
        out_69 = torch.nn.functional.batch_norm(
            out_68,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_68 = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_1_modules_bn3_parameters_bias_ = None
        out_69 += out_63
        out_70 = out_69
        out_69 = out_63 = None
        out_71 = torch.nn.functional.relu(out_70, inplace=True)
        out_70 = None
        out_72 = torch.conv2d(
            out_71,
            l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_2_modules_conv1_parameters_weight_ = None
        out_73 = torch.nn.functional.batch_norm(
            out_72,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_72 = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn1_parameters_bias_ = None
        out_74 = torch.nn.functional.relu(out_73, inplace=True)
        out_73 = None
        split_9 = torch.functional.split(out_74, 128, 1)
        out_74 = None
        sp_108 = split_9[0]
        getitem_37 = split_9[1]
        getitem_38 = split_9[2]
        getitem_39 = split_9[3]
        split_9 = None
        sp_109 = torch.conv2d(
            sp_108,
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_108 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_110 = torch.nn.functional.batch_norm(
            sp_109,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_109 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_111 = torch.nn.functional.relu(sp_110, inplace=True)
        sp_110 = None
        sp_112 = sp_111 + getitem_37
        getitem_37 = None
        sp_113 = torch.conv2d(
            sp_112,
            l_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_112 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_114 = torch.nn.functional.batch_norm(
            sp_113,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_113 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_115 = torch.nn.functional.relu(sp_114, inplace=True)
        sp_114 = None
        sp_116 = sp_115 + getitem_38
        getitem_38 = None
        sp_117 = torch.conv2d(
            sp_116,
            l_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_116 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_118 = torch.nn.functional.batch_norm(
            sp_117,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_117 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_119 = torch.nn.functional.relu(sp_118, inplace=True)
        sp_118 = None
        out_75 = torch.cat([sp_111, sp_115, sp_119, getitem_39], 1)
        sp_111 = sp_115 = sp_119 = getitem_39 = None
        out_76 = torch.conv2d(
            out_75,
            l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_75 = l_self_modules_layer3_modules_2_modules_conv3_parameters_weight_ = None
        out_77 = torch.nn.functional.batch_norm(
            out_76,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_76 = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_2_modules_bn3_parameters_bias_ = None
        out_77 += out_71
        out_78 = out_77
        out_77 = out_71 = None
        out_79 = torch.nn.functional.relu(out_78, inplace=True)
        out_78 = None
        out_80 = torch.conv2d(
            out_79,
            l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_3_modules_conv1_parameters_weight_ = None
        out_81 = torch.nn.functional.batch_norm(
            out_80,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_80 = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn1_parameters_bias_ = None
        out_82 = torch.nn.functional.relu(out_81, inplace=True)
        out_81 = None
        split_10 = torch.functional.split(out_82, 128, 1)
        out_82 = None
        sp_120 = split_10[0]
        getitem_41 = split_10[1]
        getitem_42 = split_10[2]
        getitem_43 = split_10[3]
        split_10 = None
        sp_121 = torch.conv2d(
            sp_120,
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_120 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_122 = torch.nn.functional.batch_norm(
            sp_121,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_121 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_123 = torch.nn.functional.relu(sp_122, inplace=True)
        sp_122 = None
        sp_124 = sp_123 + getitem_41
        getitem_41 = None
        sp_125 = torch.conv2d(
            sp_124,
            l_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_124 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_126 = torch.nn.functional.batch_norm(
            sp_125,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_125 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_127 = torch.nn.functional.relu(sp_126, inplace=True)
        sp_126 = None
        sp_128 = sp_127 + getitem_42
        getitem_42 = None
        sp_129 = torch.conv2d(
            sp_128,
            l_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_128 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_130 = torch.nn.functional.batch_norm(
            sp_129,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_129 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_131 = torch.nn.functional.relu(sp_130, inplace=True)
        sp_130 = None
        out_83 = torch.cat([sp_123, sp_127, sp_131, getitem_43], 1)
        sp_123 = sp_127 = sp_131 = getitem_43 = None
        out_84 = torch.conv2d(
            out_83,
            l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_83 = l_self_modules_layer3_modules_3_modules_conv3_parameters_weight_ = None
        out_85 = torch.nn.functional.batch_norm(
            out_84,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_84 = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_3_modules_bn3_parameters_bias_ = None
        out_85 += out_79
        out_86 = out_85
        out_85 = out_79 = None
        out_87 = torch.nn.functional.relu(out_86, inplace=True)
        out_86 = None
        out_88 = torch.conv2d(
            out_87,
            l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_4_modules_conv1_parameters_weight_ = None
        out_89 = torch.nn.functional.batch_norm(
            out_88,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_88 = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn1_parameters_bias_ = None
        out_90 = torch.nn.functional.relu(out_89, inplace=True)
        out_89 = None
        split_11 = torch.functional.split(out_90, 128, 1)
        out_90 = None
        sp_132 = split_11[0]
        getitem_45 = split_11[1]
        getitem_46 = split_11[2]
        getitem_47 = split_11[3]
        split_11 = None
        sp_133 = torch.conv2d(
            sp_132,
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_132 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_134 = torch.nn.functional.batch_norm(
            sp_133,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_133 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_135 = torch.nn.functional.relu(sp_134, inplace=True)
        sp_134 = None
        sp_136 = sp_135 + getitem_45
        getitem_45 = None
        sp_137 = torch.conv2d(
            sp_136,
            l_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_136 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_138 = torch.nn.functional.batch_norm(
            sp_137,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_137 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_139 = torch.nn.functional.relu(sp_138, inplace=True)
        sp_138 = None
        sp_140 = sp_139 + getitem_46
        getitem_46 = None
        sp_141 = torch.conv2d(
            sp_140,
            l_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_140 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_142 = torch.nn.functional.batch_norm(
            sp_141,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_141 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_143 = torch.nn.functional.relu(sp_142, inplace=True)
        sp_142 = None
        out_91 = torch.cat([sp_135, sp_139, sp_143, getitem_47], 1)
        sp_135 = sp_139 = sp_143 = getitem_47 = None
        out_92 = torch.conv2d(
            out_91,
            l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_91 = l_self_modules_layer3_modules_4_modules_conv3_parameters_weight_ = None
        out_93 = torch.nn.functional.batch_norm(
            out_92,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_92 = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_4_modules_bn3_parameters_bias_ = None
        out_93 += out_87
        out_94 = out_93
        out_93 = out_87 = None
        out_95 = torch.nn.functional.relu(out_94, inplace=True)
        out_94 = None
        out_96 = torch.conv2d(
            out_95,
            l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_5_modules_conv1_parameters_weight_ = None
        out_97 = torch.nn.functional.batch_norm(
            out_96,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_96 = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn1_parameters_bias_ = None
        out_98 = torch.nn.functional.relu(out_97, inplace=True)
        out_97 = None
        split_12 = torch.functional.split(out_98, 128, 1)
        out_98 = None
        sp_144 = split_12[0]
        getitem_49 = split_12[1]
        getitem_50 = split_12[2]
        getitem_51 = split_12[3]
        split_12 = None
        sp_145 = torch.conv2d(
            sp_144,
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_144 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_146 = torch.nn.functional.batch_norm(
            sp_145,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_145 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_147 = torch.nn.functional.relu(sp_146, inplace=True)
        sp_146 = None
        sp_148 = sp_147 + getitem_49
        getitem_49 = None
        sp_149 = torch.conv2d(
            sp_148,
            l_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_148 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_150 = torch.nn.functional.batch_norm(
            sp_149,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_149 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_151 = torch.nn.functional.relu(sp_150, inplace=True)
        sp_150 = None
        sp_152 = sp_151 + getitem_50
        getitem_50 = None
        sp_153 = torch.conv2d(
            sp_152,
            l_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_152 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_154 = torch.nn.functional.batch_norm(
            sp_153,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_153 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_155 = torch.nn.functional.relu(sp_154, inplace=True)
        sp_154 = None
        out_99 = torch.cat([sp_147, sp_151, sp_155, getitem_51], 1)
        sp_147 = sp_151 = sp_155 = getitem_51 = None
        out_100 = torch.conv2d(
            out_99,
            l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_99 = l_self_modules_layer3_modules_5_modules_conv3_parameters_weight_ = None
        out_101 = torch.nn.functional.batch_norm(
            out_100,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_100 = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_5_modules_bn3_parameters_bias_ = None
        out_101 += out_95
        out_102 = out_101
        out_101 = out_95 = None
        out_103 = torch.nn.functional.relu(out_102, inplace=True)
        out_102 = None
        out_104 = torch.conv2d(
            out_103,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        out_105 = torch.nn.functional.batch_norm(
            out_104,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_104 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        out_106 = torch.nn.functional.relu(out_105, inplace=True)
        out_105 = None
        split_13 = torch.functional.split(out_106, 256, 1)
        out_106 = None
        sp_156 = split_13[0]
        sp_160 = split_13[1]
        sp_164 = split_13[2]
        getitem_55 = split_13[3]
        split_13 = None
        sp_157 = torch.conv2d(
            sp_156,
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_156 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_158 = torch.nn.functional.batch_norm(
            sp_157,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_157 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_159 = torch.nn.functional.relu(sp_158, inplace=True)
        sp_158 = None
        sp_161 = torch.conv2d(
            sp_160,
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_160 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_162 = torch.nn.functional.batch_norm(
            sp_161,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_161 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_163 = torch.nn.functional.relu(sp_162, inplace=True)
        sp_162 = None
        sp_165 = torch.conv2d(
            sp_164,
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            8,
        )
        sp_164 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_166 = torch.nn.functional.batch_norm(
            sp_165,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_165 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_167 = torch.nn.functional.relu(sp_166, inplace=True)
        sp_166 = None
        avg_pool2d_3 = torch._C._nn.avg_pool2d(getitem_55, 3, 2, 1, False, True, None)
        getitem_55 = None
        out_107 = torch.cat([sp_159, sp_163, sp_167, avg_pool2d_3], 1)
        sp_159 = sp_163 = sp_167 = avg_pool2d_3 = None
        out_108 = torch.conv2d(
            out_107,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_107 = (
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_109 = torch.nn.functional.batch_norm(
            out_108,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_108 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        input_7 = torch.conv2d(
            out_103,
            l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_,
            None,
            (2, 2),
            (0, 0),
            (1, 1),
            1,
        )
        out_103 = l_self_modules_layer4_modules_0_modules_downsample_modules_0_parameters_weight_ = (None)
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
        out_109 += input_8
        out_110 = out_109
        out_109 = input_8 = None
        out_111 = torch.nn.functional.relu(out_110, inplace=True)
        out_110 = None
        out_112 = torch.conv2d(
            out_111,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        out_113 = torch.nn.functional.batch_norm(
            out_112,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_112 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        out_114 = torch.nn.functional.relu(out_113, inplace=True)
        out_113 = None
        split_14 = torch.functional.split(out_114, 256, 1)
        out_114 = None
        sp_168 = split_14[0]
        getitem_57 = split_14[1]
        getitem_58 = split_14[2]
        getitem_59 = split_14[3]
        split_14 = None
        sp_169 = torch.conv2d(
            sp_168,
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_168 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_170 = torch.nn.functional.batch_norm(
            sp_169,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_169 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_171 = torch.nn.functional.relu(sp_170, inplace=True)
        sp_170 = None
        sp_172 = sp_171 + getitem_57
        getitem_57 = None
        sp_173 = torch.conv2d(
            sp_172,
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_172 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_174 = torch.nn.functional.batch_norm(
            sp_173,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_173 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_175 = torch.nn.functional.relu(sp_174, inplace=True)
        sp_174 = None
        sp_176 = sp_175 + getitem_58
        getitem_58 = None
        sp_177 = torch.conv2d(
            sp_176,
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_176 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_178 = torch.nn.functional.batch_norm(
            sp_177,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_177 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_179 = torch.nn.functional.relu(sp_178, inplace=True)
        sp_178 = None
        out_115 = torch.cat([sp_171, sp_175, sp_179, getitem_59], 1)
        sp_171 = sp_175 = sp_179 = getitem_59 = None
        out_116 = torch.conv2d(
            out_115,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_115 = (
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        out_117 += out_111
        out_118 = out_117
        out_117 = out_111 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_120 = torch.conv2d(
            out_119,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        split_15 = torch.functional.split(out_122, 256, 1)
        out_122 = None
        sp_180 = split_15[0]
        getitem_61 = split_15[1]
        getitem_62 = split_15[2]
        getitem_63 = split_15[3]
        split_15 = None
        sp_181 = torch.conv2d(
            sp_180,
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_180 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_182 = torch.nn.functional.batch_norm(
            sp_181,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_181 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_183 = torch.nn.functional.relu(sp_182, inplace=True)
        sp_182 = None
        sp_184 = sp_183 + getitem_61
        getitem_61 = None
        sp_185 = torch.conv2d(
            sp_184,
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_184 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_186 = torch.nn.functional.batch_norm(
            sp_185,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_185 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_187 = torch.nn.functional.relu(sp_186, inplace=True)
        sp_186 = None
        sp_188 = sp_187 + getitem_62
        getitem_62 = None
        sp_189 = torch.conv2d(
            sp_188,
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            8,
        )
        sp_188 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_190 = torch.nn.functional.batch_norm(
            sp_189,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_189 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_191 = torch.nn.functional.relu(sp_190, inplace=True)
        sp_190 = None
        out_123 = torch.cat([sp_183, sp_187, sp_191, getitem_63], 1)
        sp_183 = sp_187 = sp_191 = getitem_63 = None
        out_124 = torch.conv2d(
            out_123,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_123 = (
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_125 = torch.nn.functional.batch_norm(
            out_124,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_124 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        out_125 += out_119
        out_126 = out_125
        out_125 = out_119 = None
        out_127 = torch.nn.functional.relu(out_126, inplace=True)
        out_126 = None
        x_4 = torch.nn.functional.adaptive_avg_pool2d(out_127, 1)
        out_127 = None
        x_5 = x_4.flatten(1, -1)
        x_4 = None
        x_6 = torch._C._nn.linear(
            x_5,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_5 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_6,)
