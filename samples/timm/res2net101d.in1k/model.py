import torch


class GraphModule(torch.nn.Module):
    def forward(
        self,
        L_self_modules_conv1_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        s1: torch.SymInt,
        L_x_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_conv1_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_conv1_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_conv1_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_6_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_6_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_6_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_7_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_7_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_8_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_8_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_9_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_9_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_10_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_10_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_11_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_11_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_12_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_12_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_13_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_13_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_14_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_14_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_15_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_15_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_16_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_16_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_17_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_17_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_18_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_18_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_19_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_19_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_20_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_20_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_21_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_21_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_conv1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_conv3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_22_modules_bn3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_22_modules_bn3_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_conv1_modules_0_parameters_weight_ = (
            L_self_modules_conv1_modules_0_parameters_weight_
        )
        l_x_ = L_x_
        l_self_modules_conv1_modules_1_buffers_running_mean_ = (
            L_self_modules_conv1_modules_1_buffers_running_mean_
        )
        l_self_modules_conv1_modules_1_buffers_running_var_ = (
            L_self_modules_conv1_modules_1_buffers_running_var_
        )
        l_self_modules_conv1_modules_1_parameters_weight_ = (
            L_self_modules_conv1_modules_1_parameters_weight_
        )
        l_self_modules_conv1_modules_1_parameters_bias_ = (
            L_self_modules_conv1_modules_1_parameters_bias_
        )
        l_self_modules_conv1_modules_3_parameters_weight_ = (
            L_self_modules_conv1_modules_3_parameters_weight_
        )
        l_self_modules_conv1_modules_4_buffers_running_mean_ = (
            L_self_modules_conv1_modules_4_buffers_running_mean_
        )
        l_self_modules_conv1_modules_4_buffers_running_var_ = (
            L_self_modules_conv1_modules_4_buffers_running_var_
        )
        l_self_modules_conv1_modules_4_parameters_weight_ = (
            L_self_modules_conv1_modules_4_parameters_weight_
        )
        l_self_modules_conv1_modules_4_parameters_bias_ = (
            L_self_modules_conv1_modules_4_parameters_bias_
        )
        l_self_modules_conv1_modules_6_parameters_weight_ = (
            L_self_modules_conv1_modules_6_parameters_weight_
        )
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
        l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_
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
        l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_
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
        l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_
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
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_6_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_7_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_8_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_9_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_10_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_11_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_12_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_13_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_14_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_15_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_16_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_17_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_18_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_19_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_20_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_21_modules_bn3_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_conv1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bn1_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_
        )
        l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_conv3_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_ = (
            L_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        )
        l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_ = (
            L_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        )
        l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = (
            L_self_modules_layer3_modules_22_modules_bn3_parameters_bias_
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
        l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_
        l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = L_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_
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
        input_1 = torch.conv2d(
            l_x_,
            l_self_modules_conv1_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        l_x_ = l_self_modules_conv1_modules_0_parameters_weight_ = None
        input_2 = torch.nn.functional.batch_norm(
            input_1,
            l_self_modules_conv1_modules_1_buffers_running_mean_,
            l_self_modules_conv1_modules_1_buffers_running_var_,
            l_self_modules_conv1_modules_1_parameters_weight_,
            l_self_modules_conv1_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_1 = (
            l_self_modules_conv1_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_1_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_1_parameters_weight_
        ) = l_self_modules_conv1_modules_1_parameters_bias_ = None
        input_3 = torch.nn.functional.relu(input_2, inplace=True)
        input_2 = None
        input_4 = torch.conv2d(
            input_3,
            l_self_modules_conv1_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_3 = l_self_modules_conv1_modules_3_parameters_weight_ = None
        input_5 = torch.nn.functional.batch_norm(
            input_4,
            l_self_modules_conv1_modules_4_buffers_running_mean_,
            l_self_modules_conv1_modules_4_buffers_running_var_,
            l_self_modules_conv1_modules_4_parameters_weight_,
            l_self_modules_conv1_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_4 = (
            l_self_modules_conv1_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_conv1_modules_4_buffers_running_var_
        ) = (
            l_self_modules_conv1_modules_4_parameters_weight_
        ) = l_self_modules_conv1_modules_4_parameters_bias_ = None
        input_6 = torch.nn.functional.relu(input_5, inplace=True)
        input_5 = None
        input_7 = torch.conv2d(
            input_6,
            l_self_modules_conv1_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        input_6 = l_self_modules_conv1_modules_6_parameters_weight_ = None
        x = torch.nn.functional.batch_norm(
            input_7,
            l_self_modules_bn1_buffers_running_mean_,
            l_self_modules_bn1_buffers_running_var_,
            l_self_modules_bn1_parameters_weight_,
            l_self_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_7 = (
            l_self_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_bn1_parameters_weight_
        ) = l_self_modules_bn1_parameters_bias_ = None
        x_1 = torch.nn.functional.relu(x, inplace=True)
        x = None
        x_2 = torch.nn.functional.max_pool2d(
            x_1, 3, 2, 1, 1, ceil_mode=False, return_indices=False
        )
        x_1 = None
        out = torch.conv2d(
            x_2,
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
        split = torch.functional.split(out_2, 26, 1)
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
            1,
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
            1,
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
            1,
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
        input_8 = torch.conv2d(
            x_2,
            l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        x_2 = l_self_modules_layer1_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_9 = torch.nn.functional.batch_norm(
            input_8,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_8 = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer1_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_5 += input_9
        out_6 = out_5
        out_5 = input_9 = None
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
        split_1 = torch.functional.split(out_10, 26, 1)
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
            1,
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
            1,
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
            1,
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
        split_2 = torch.functional.split(out_18, 26, 1)
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
            1,
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
            1,
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
            1,
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
        split_3 = torch.functional.split(out_26, 52, 1)
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
            1,
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
            1,
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
            1,
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
        input_10 = torch._C._nn.avg_pool2d(out_23, 2, 2, 0, True, False, None)
        out_23 = None
        input_11 = torch.conv2d(
            input_10,
            l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_10 = l_self_modules_layer2_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_12 = torch.nn.functional.batch_norm(
            input_11,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_11 = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer2_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_29 += input_12
        out_30 = out_29
        out_29 = input_12 = None
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
        split_4 = torch.functional.split(out_34, 52, 1)
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
            1,
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
            1,
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
            1,
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
        split_5 = torch.functional.split(out_42, 52, 1)
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
            1,
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
            1,
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
            1,
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
        split_6 = torch.functional.split(out_50, 52, 1)
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
            1,
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
            1,
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
            1,
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
        split_7 = torch.functional.split(out_58, 104, 1)
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
            1,
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
            1,
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
            1,
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
        avg_pool2d_3 = torch._C._nn.avg_pool2d(getitem_31, 3, 2, 1, False, True, None)
        getitem_31 = None
        out_59 = torch.cat([sp_87, sp_91, sp_95, avg_pool2d_3], 1)
        sp_87 = sp_91 = sp_95 = avg_pool2d_3 = None
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
        input_13 = torch._C._nn.avg_pool2d(out_55, 2, 2, 0, True, False, None)
        out_55 = None
        input_14 = torch.conv2d(
            input_13,
            l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_13 = l_self_modules_layer3_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_15 = torch.nn.functional.batch_norm(
            input_14,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_14 = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer3_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_61 += input_15
        out_62 = out_61
        out_61 = input_15 = None
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
        split_8 = torch.functional.split(out_66, 104, 1)
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
            1,
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
            1,
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
            1,
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
        split_9 = torch.functional.split(out_74, 104, 1)
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
            1,
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
            1,
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
            1,
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
        split_10 = torch.functional.split(out_82, 104, 1)
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
            1,
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
            1,
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
            1,
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
        split_11 = torch.functional.split(out_90, 104, 1)
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
            1,
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
            1,
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
            1,
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
        split_12 = torch.functional.split(out_98, 104, 1)
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
            1,
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
            1,
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
            1,
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
            l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_6_modules_conv1_parameters_weight_ = None
        out_105 = torch.nn.functional.batch_norm(
            out_104,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_104 = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn1_parameters_bias_ = None
        out_106 = torch.nn.functional.relu(out_105, inplace=True)
        out_105 = None
        split_13 = torch.functional.split(out_106, 104, 1)
        out_106 = None
        sp_156 = split_13[0]
        getitem_53 = split_13[1]
        getitem_54 = split_13[2]
        getitem_55 = split_13[3]
        split_13 = None
        sp_157 = torch.conv2d(
            sp_156,
            l_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_156 = (
            l_self_modules_layer3_modules_6_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_158 = torch.nn.functional.batch_norm(
            sp_157,
            l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_157 = (
            l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_159 = torch.nn.functional.relu(sp_158, inplace=True)
        sp_158 = None
        sp_160 = sp_159 + getitem_53
        getitem_53 = None
        sp_161 = torch.conv2d(
            sp_160,
            l_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_160 = (
            l_self_modules_layer3_modules_6_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_162 = torch.nn.functional.batch_norm(
            sp_161,
            l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_161 = (
            l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_163 = torch.nn.functional.relu(sp_162, inplace=True)
        sp_162 = None
        sp_164 = sp_163 + getitem_54
        getitem_54 = None
        sp_165 = torch.conv2d(
            sp_164,
            l_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_164 = (
            l_self_modules_layer3_modules_6_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_166 = torch.nn.functional.batch_norm(
            sp_165,
            l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_165 = (
            l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_6_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_167 = torch.nn.functional.relu(sp_166, inplace=True)
        sp_166 = None
        out_107 = torch.cat([sp_159, sp_163, sp_167, getitem_55], 1)
        sp_159 = sp_163 = sp_167 = getitem_55 = None
        out_108 = torch.conv2d(
            out_107,
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_107 = (
            l_self_modules_layer3_modules_6_modules_conv3_parameters_weight_
        ) = None
        out_109 = torch.nn.functional.batch_norm(
            out_108,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_108 = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_6_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_6_modules_bn3_parameters_bias_ = None
        out_109 += out_103
        out_110 = out_109
        out_109 = out_103 = None
        out_111 = torch.nn.functional.relu(out_110, inplace=True)
        out_110 = None
        out_112 = torch.conv2d(
            out_111,
            l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_7_modules_conv1_parameters_weight_ = None
        out_113 = torch.nn.functional.batch_norm(
            out_112,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_112 = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn1_parameters_bias_ = None
        out_114 = torch.nn.functional.relu(out_113, inplace=True)
        out_113 = None
        split_14 = torch.functional.split(out_114, 104, 1)
        out_114 = None
        sp_168 = split_14[0]
        getitem_57 = split_14[1]
        getitem_58 = split_14[2]
        getitem_59 = split_14[3]
        split_14 = None
        sp_169 = torch.conv2d(
            sp_168,
            l_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_168 = (
            l_self_modules_layer3_modules_7_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_170 = torch.nn.functional.batch_norm(
            sp_169,
            l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_169 = (
            l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_171 = torch.nn.functional.relu(sp_170, inplace=True)
        sp_170 = None
        sp_172 = sp_171 + getitem_57
        getitem_57 = None
        sp_173 = torch.conv2d(
            sp_172,
            l_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_172 = (
            l_self_modules_layer3_modules_7_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_174 = torch.nn.functional.batch_norm(
            sp_173,
            l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_173 = (
            l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_175 = torch.nn.functional.relu(sp_174, inplace=True)
        sp_174 = None
        sp_176 = sp_175 + getitem_58
        getitem_58 = None
        sp_177 = torch.conv2d(
            sp_176,
            l_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_176 = (
            l_self_modules_layer3_modules_7_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_178 = torch.nn.functional.batch_norm(
            sp_177,
            l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_177 = (
            l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_7_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_179 = torch.nn.functional.relu(sp_178, inplace=True)
        sp_178 = None
        out_115 = torch.cat([sp_171, sp_175, sp_179, getitem_59], 1)
        sp_171 = sp_175 = sp_179 = getitem_59 = None
        out_116 = torch.conv2d(
            out_115,
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_115 = (
            l_self_modules_layer3_modules_7_modules_conv3_parameters_weight_
        ) = None
        out_117 = torch.nn.functional.batch_norm(
            out_116,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_116 = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_7_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_7_modules_bn3_parameters_bias_ = None
        out_117 += out_111
        out_118 = out_117
        out_117 = out_111 = None
        out_119 = torch.nn.functional.relu(out_118, inplace=True)
        out_118 = None
        out_120 = torch.conv2d(
            out_119,
            l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_8_modules_conv1_parameters_weight_ = None
        out_121 = torch.nn.functional.batch_norm(
            out_120,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_120 = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn1_parameters_bias_ = None
        out_122 = torch.nn.functional.relu(out_121, inplace=True)
        out_121 = None
        split_15 = torch.functional.split(out_122, 104, 1)
        out_122 = None
        sp_180 = split_15[0]
        getitem_61 = split_15[1]
        getitem_62 = split_15[2]
        getitem_63 = split_15[3]
        split_15 = None
        sp_181 = torch.conv2d(
            sp_180,
            l_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_180 = (
            l_self_modules_layer3_modules_8_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_182 = torch.nn.functional.batch_norm(
            sp_181,
            l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_181 = (
            l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_183 = torch.nn.functional.relu(sp_182, inplace=True)
        sp_182 = None
        sp_184 = sp_183 + getitem_61
        getitem_61 = None
        sp_185 = torch.conv2d(
            sp_184,
            l_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_184 = (
            l_self_modules_layer3_modules_8_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_186 = torch.nn.functional.batch_norm(
            sp_185,
            l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_185 = (
            l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_187 = torch.nn.functional.relu(sp_186, inplace=True)
        sp_186 = None
        sp_188 = sp_187 + getitem_62
        getitem_62 = None
        sp_189 = torch.conv2d(
            sp_188,
            l_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_188 = (
            l_self_modules_layer3_modules_8_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_190 = torch.nn.functional.batch_norm(
            sp_189,
            l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_189 = (
            l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_8_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_191 = torch.nn.functional.relu(sp_190, inplace=True)
        sp_190 = None
        out_123 = torch.cat([sp_183, sp_187, sp_191, getitem_63], 1)
        sp_183 = sp_187 = sp_191 = getitem_63 = None
        out_124 = torch.conv2d(
            out_123,
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_123 = (
            l_self_modules_layer3_modules_8_modules_conv3_parameters_weight_
        ) = None
        out_125 = torch.nn.functional.batch_norm(
            out_124,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_124 = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_8_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_8_modules_bn3_parameters_bias_ = None
        out_125 += out_119
        out_126 = out_125
        out_125 = out_119 = None
        out_127 = torch.nn.functional.relu(out_126, inplace=True)
        out_126 = None
        out_128 = torch.conv2d(
            out_127,
            l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_9_modules_conv1_parameters_weight_ = None
        out_129 = torch.nn.functional.batch_norm(
            out_128,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_128 = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn1_parameters_bias_ = None
        out_130 = torch.nn.functional.relu(out_129, inplace=True)
        out_129 = None
        split_16 = torch.functional.split(out_130, 104, 1)
        out_130 = None
        sp_192 = split_16[0]
        getitem_65 = split_16[1]
        getitem_66 = split_16[2]
        getitem_67 = split_16[3]
        split_16 = None
        sp_193 = torch.conv2d(
            sp_192,
            l_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_192 = (
            l_self_modules_layer3_modules_9_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_194 = torch.nn.functional.batch_norm(
            sp_193,
            l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_193 = (
            l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_195 = torch.nn.functional.relu(sp_194, inplace=True)
        sp_194 = None
        sp_196 = sp_195 + getitem_65
        getitem_65 = None
        sp_197 = torch.conv2d(
            sp_196,
            l_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_196 = (
            l_self_modules_layer3_modules_9_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_198 = torch.nn.functional.batch_norm(
            sp_197,
            l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_197 = (
            l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_199 = torch.nn.functional.relu(sp_198, inplace=True)
        sp_198 = None
        sp_200 = sp_199 + getitem_66
        getitem_66 = None
        sp_201 = torch.conv2d(
            sp_200,
            l_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_200 = (
            l_self_modules_layer3_modules_9_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_202 = torch.nn.functional.batch_norm(
            sp_201,
            l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_201 = (
            l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_9_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_203 = torch.nn.functional.relu(sp_202, inplace=True)
        sp_202 = None
        out_131 = torch.cat([sp_195, sp_199, sp_203, getitem_67], 1)
        sp_195 = sp_199 = sp_203 = getitem_67 = None
        out_132 = torch.conv2d(
            out_131,
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_131 = (
            l_self_modules_layer3_modules_9_modules_conv3_parameters_weight_
        ) = None
        out_133 = torch.nn.functional.batch_norm(
            out_132,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_132 = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_9_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_9_modules_bn3_parameters_bias_ = None
        out_133 += out_127
        out_134 = out_133
        out_133 = out_127 = None
        out_135 = torch.nn.functional.relu(out_134, inplace=True)
        out_134 = None
        out_136 = torch.conv2d(
            out_135,
            l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_10_modules_conv1_parameters_weight_ = None
        out_137 = torch.nn.functional.batch_norm(
            out_136,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_136 = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn1_parameters_bias_ = None
        out_138 = torch.nn.functional.relu(out_137, inplace=True)
        out_137 = None
        split_17 = torch.functional.split(out_138, 104, 1)
        out_138 = None
        sp_204 = split_17[0]
        getitem_69 = split_17[1]
        getitem_70 = split_17[2]
        getitem_71 = split_17[3]
        split_17 = None
        sp_205 = torch.conv2d(
            sp_204,
            l_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_204 = (
            l_self_modules_layer3_modules_10_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_206 = torch.nn.functional.batch_norm(
            sp_205,
            l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_205 = (
            l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_207 = torch.nn.functional.relu(sp_206, inplace=True)
        sp_206 = None
        sp_208 = sp_207 + getitem_69
        getitem_69 = None
        sp_209 = torch.conv2d(
            sp_208,
            l_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_208 = (
            l_self_modules_layer3_modules_10_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_210 = torch.nn.functional.batch_norm(
            sp_209,
            l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_209 = (
            l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_211 = torch.nn.functional.relu(sp_210, inplace=True)
        sp_210 = None
        sp_212 = sp_211 + getitem_70
        getitem_70 = None
        sp_213 = torch.conv2d(
            sp_212,
            l_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_212 = (
            l_self_modules_layer3_modules_10_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_214 = torch.nn.functional.batch_norm(
            sp_213,
            l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_213 = (
            l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_10_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_215 = torch.nn.functional.relu(sp_214, inplace=True)
        sp_214 = None
        out_139 = torch.cat([sp_207, sp_211, sp_215, getitem_71], 1)
        sp_207 = sp_211 = sp_215 = getitem_71 = None
        out_140 = torch.conv2d(
            out_139,
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_139 = (
            l_self_modules_layer3_modules_10_modules_conv3_parameters_weight_
        ) = None
        out_141 = torch.nn.functional.batch_norm(
            out_140,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_140 = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_10_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_10_modules_bn3_parameters_bias_ = None
        out_141 += out_135
        out_142 = out_141
        out_141 = out_135 = None
        out_143 = torch.nn.functional.relu(out_142, inplace=True)
        out_142 = None
        out_144 = torch.conv2d(
            out_143,
            l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_11_modules_conv1_parameters_weight_ = None
        out_145 = torch.nn.functional.batch_norm(
            out_144,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_144 = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn1_parameters_bias_ = None
        out_146 = torch.nn.functional.relu(out_145, inplace=True)
        out_145 = None
        split_18 = torch.functional.split(out_146, 104, 1)
        out_146 = None
        sp_216 = split_18[0]
        getitem_73 = split_18[1]
        getitem_74 = split_18[2]
        getitem_75 = split_18[3]
        split_18 = None
        sp_217 = torch.conv2d(
            sp_216,
            l_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_216 = (
            l_self_modules_layer3_modules_11_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_218 = torch.nn.functional.batch_norm(
            sp_217,
            l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_217 = (
            l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_219 = torch.nn.functional.relu(sp_218, inplace=True)
        sp_218 = None
        sp_220 = sp_219 + getitem_73
        getitem_73 = None
        sp_221 = torch.conv2d(
            sp_220,
            l_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_220 = (
            l_self_modules_layer3_modules_11_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_222 = torch.nn.functional.batch_norm(
            sp_221,
            l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_221 = (
            l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_223 = torch.nn.functional.relu(sp_222, inplace=True)
        sp_222 = None
        sp_224 = sp_223 + getitem_74
        getitem_74 = None
        sp_225 = torch.conv2d(
            sp_224,
            l_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_224 = (
            l_self_modules_layer3_modules_11_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_226 = torch.nn.functional.batch_norm(
            sp_225,
            l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_225 = (
            l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_11_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_227 = torch.nn.functional.relu(sp_226, inplace=True)
        sp_226 = None
        out_147 = torch.cat([sp_219, sp_223, sp_227, getitem_75], 1)
        sp_219 = sp_223 = sp_227 = getitem_75 = None
        out_148 = torch.conv2d(
            out_147,
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_147 = (
            l_self_modules_layer3_modules_11_modules_conv3_parameters_weight_
        ) = None
        out_149 = torch.nn.functional.batch_norm(
            out_148,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_148 = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_11_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_11_modules_bn3_parameters_bias_ = None
        out_149 += out_143
        out_150 = out_149
        out_149 = out_143 = None
        out_151 = torch.nn.functional.relu(out_150, inplace=True)
        out_150 = None
        out_152 = torch.conv2d(
            out_151,
            l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_12_modules_conv1_parameters_weight_ = None
        out_153 = torch.nn.functional.batch_norm(
            out_152,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_152 = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn1_parameters_bias_ = None
        out_154 = torch.nn.functional.relu(out_153, inplace=True)
        out_153 = None
        split_19 = torch.functional.split(out_154, 104, 1)
        out_154 = None
        sp_228 = split_19[0]
        getitem_77 = split_19[1]
        getitem_78 = split_19[2]
        getitem_79 = split_19[3]
        split_19 = None
        sp_229 = torch.conv2d(
            sp_228,
            l_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_228 = (
            l_self_modules_layer3_modules_12_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_230 = torch.nn.functional.batch_norm(
            sp_229,
            l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_229 = (
            l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_231 = torch.nn.functional.relu(sp_230, inplace=True)
        sp_230 = None
        sp_232 = sp_231 + getitem_77
        getitem_77 = None
        sp_233 = torch.conv2d(
            sp_232,
            l_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_232 = (
            l_self_modules_layer3_modules_12_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_234 = torch.nn.functional.batch_norm(
            sp_233,
            l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_233 = (
            l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_235 = torch.nn.functional.relu(sp_234, inplace=True)
        sp_234 = None
        sp_236 = sp_235 + getitem_78
        getitem_78 = None
        sp_237 = torch.conv2d(
            sp_236,
            l_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_236 = (
            l_self_modules_layer3_modules_12_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_238 = torch.nn.functional.batch_norm(
            sp_237,
            l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_237 = (
            l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_12_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_239 = torch.nn.functional.relu(sp_238, inplace=True)
        sp_238 = None
        out_155 = torch.cat([sp_231, sp_235, sp_239, getitem_79], 1)
        sp_231 = sp_235 = sp_239 = getitem_79 = None
        out_156 = torch.conv2d(
            out_155,
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_155 = (
            l_self_modules_layer3_modules_12_modules_conv3_parameters_weight_
        ) = None
        out_157 = torch.nn.functional.batch_norm(
            out_156,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_156 = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_12_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_12_modules_bn3_parameters_bias_ = None
        out_157 += out_151
        out_158 = out_157
        out_157 = out_151 = None
        out_159 = torch.nn.functional.relu(out_158, inplace=True)
        out_158 = None
        out_160 = torch.conv2d(
            out_159,
            l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_13_modules_conv1_parameters_weight_ = None
        out_161 = torch.nn.functional.batch_norm(
            out_160,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_160 = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn1_parameters_bias_ = None
        out_162 = torch.nn.functional.relu(out_161, inplace=True)
        out_161 = None
        split_20 = torch.functional.split(out_162, 104, 1)
        out_162 = None
        sp_240 = split_20[0]
        getitem_81 = split_20[1]
        getitem_82 = split_20[2]
        getitem_83 = split_20[3]
        split_20 = None
        sp_241 = torch.conv2d(
            sp_240,
            l_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_240 = (
            l_self_modules_layer3_modules_13_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_242 = torch.nn.functional.batch_norm(
            sp_241,
            l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_241 = (
            l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_243 = torch.nn.functional.relu(sp_242, inplace=True)
        sp_242 = None
        sp_244 = sp_243 + getitem_81
        getitem_81 = None
        sp_245 = torch.conv2d(
            sp_244,
            l_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_244 = (
            l_self_modules_layer3_modules_13_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_246 = torch.nn.functional.batch_norm(
            sp_245,
            l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_245 = (
            l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_247 = torch.nn.functional.relu(sp_246, inplace=True)
        sp_246 = None
        sp_248 = sp_247 + getitem_82
        getitem_82 = None
        sp_249 = torch.conv2d(
            sp_248,
            l_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_248 = (
            l_self_modules_layer3_modules_13_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_250 = torch.nn.functional.batch_norm(
            sp_249,
            l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_249 = (
            l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_13_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_251 = torch.nn.functional.relu(sp_250, inplace=True)
        sp_250 = None
        out_163 = torch.cat([sp_243, sp_247, sp_251, getitem_83], 1)
        sp_243 = sp_247 = sp_251 = getitem_83 = None
        out_164 = torch.conv2d(
            out_163,
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_163 = (
            l_self_modules_layer3_modules_13_modules_conv3_parameters_weight_
        ) = None
        out_165 = torch.nn.functional.batch_norm(
            out_164,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_164 = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_13_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_13_modules_bn3_parameters_bias_ = None
        out_165 += out_159
        out_166 = out_165
        out_165 = out_159 = None
        out_167 = torch.nn.functional.relu(out_166, inplace=True)
        out_166 = None
        out_168 = torch.conv2d(
            out_167,
            l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_14_modules_conv1_parameters_weight_ = None
        out_169 = torch.nn.functional.batch_norm(
            out_168,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_168 = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn1_parameters_bias_ = None
        out_170 = torch.nn.functional.relu(out_169, inplace=True)
        out_169 = None
        split_21 = torch.functional.split(out_170, 104, 1)
        out_170 = None
        sp_252 = split_21[0]
        getitem_85 = split_21[1]
        getitem_86 = split_21[2]
        getitem_87 = split_21[3]
        split_21 = None
        sp_253 = torch.conv2d(
            sp_252,
            l_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_252 = (
            l_self_modules_layer3_modules_14_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_254 = torch.nn.functional.batch_norm(
            sp_253,
            l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_253 = (
            l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_255 = torch.nn.functional.relu(sp_254, inplace=True)
        sp_254 = None
        sp_256 = sp_255 + getitem_85
        getitem_85 = None
        sp_257 = torch.conv2d(
            sp_256,
            l_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_256 = (
            l_self_modules_layer3_modules_14_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_258 = torch.nn.functional.batch_norm(
            sp_257,
            l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_257 = (
            l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_259 = torch.nn.functional.relu(sp_258, inplace=True)
        sp_258 = None
        sp_260 = sp_259 + getitem_86
        getitem_86 = None
        sp_261 = torch.conv2d(
            sp_260,
            l_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_260 = (
            l_self_modules_layer3_modules_14_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_262 = torch.nn.functional.batch_norm(
            sp_261,
            l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_261 = (
            l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_14_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_263 = torch.nn.functional.relu(sp_262, inplace=True)
        sp_262 = None
        out_171 = torch.cat([sp_255, sp_259, sp_263, getitem_87], 1)
        sp_255 = sp_259 = sp_263 = getitem_87 = None
        out_172 = torch.conv2d(
            out_171,
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_171 = (
            l_self_modules_layer3_modules_14_modules_conv3_parameters_weight_
        ) = None
        out_173 = torch.nn.functional.batch_norm(
            out_172,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_172 = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_14_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_14_modules_bn3_parameters_bias_ = None
        out_173 += out_167
        out_174 = out_173
        out_173 = out_167 = None
        out_175 = torch.nn.functional.relu(out_174, inplace=True)
        out_174 = None
        out_176 = torch.conv2d(
            out_175,
            l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_15_modules_conv1_parameters_weight_ = None
        out_177 = torch.nn.functional.batch_norm(
            out_176,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_176 = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn1_parameters_bias_ = None
        out_178 = torch.nn.functional.relu(out_177, inplace=True)
        out_177 = None
        split_22 = torch.functional.split(out_178, 104, 1)
        out_178 = None
        sp_264 = split_22[0]
        getitem_89 = split_22[1]
        getitem_90 = split_22[2]
        getitem_91 = split_22[3]
        split_22 = None
        sp_265 = torch.conv2d(
            sp_264,
            l_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_264 = (
            l_self_modules_layer3_modules_15_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_266 = torch.nn.functional.batch_norm(
            sp_265,
            l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_265 = (
            l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_267 = torch.nn.functional.relu(sp_266, inplace=True)
        sp_266 = None
        sp_268 = sp_267 + getitem_89
        getitem_89 = None
        sp_269 = torch.conv2d(
            sp_268,
            l_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_268 = (
            l_self_modules_layer3_modules_15_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_270 = torch.nn.functional.batch_norm(
            sp_269,
            l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_269 = (
            l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_271 = torch.nn.functional.relu(sp_270, inplace=True)
        sp_270 = None
        sp_272 = sp_271 + getitem_90
        getitem_90 = None
        sp_273 = torch.conv2d(
            sp_272,
            l_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_272 = (
            l_self_modules_layer3_modules_15_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_274 = torch.nn.functional.batch_norm(
            sp_273,
            l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_273 = (
            l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_15_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_275 = torch.nn.functional.relu(sp_274, inplace=True)
        sp_274 = None
        out_179 = torch.cat([sp_267, sp_271, sp_275, getitem_91], 1)
        sp_267 = sp_271 = sp_275 = getitem_91 = None
        out_180 = torch.conv2d(
            out_179,
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_179 = (
            l_self_modules_layer3_modules_15_modules_conv3_parameters_weight_
        ) = None
        out_181 = torch.nn.functional.batch_norm(
            out_180,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_180 = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_15_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_15_modules_bn3_parameters_bias_ = None
        out_181 += out_175
        out_182 = out_181
        out_181 = out_175 = None
        out_183 = torch.nn.functional.relu(out_182, inplace=True)
        out_182 = None
        out_184 = torch.conv2d(
            out_183,
            l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_16_modules_conv1_parameters_weight_ = None
        out_185 = torch.nn.functional.batch_norm(
            out_184,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_184 = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn1_parameters_bias_ = None
        out_186 = torch.nn.functional.relu(out_185, inplace=True)
        out_185 = None
        split_23 = torch.functional.split(out_186, 104, 1)
        out_186 = None
        sp_276 = split_23[0]
        getitem_93 = split_23[1]
        getitem_94 = split_23[2]
        getitem_95 = split_23[3]
        split_23 = None
        sp_277 = torch.conv2d(
            sp_276,
            l_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_276 = (
            l_self_modules_layer3_modules_16_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_278 = torch.nn.functional.batch_norm(
            sp_277,
            l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_277 = (
            l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_279 = torch.nn.functional.relu(sp_278, inplace=True)
        sp_278 = None
        sp_280 = sp_279 + getitem_93
        getitem_93 = None
        sp_281 = torch.conv2d(
            sp_280,
            l_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_280 = (
            l_self_modules_layer3_modules_16_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_282 = torch.nn.functional.batch_norm(
            sp_281,
            l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_281 = (
            l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_283 = torch.nn.functional.relu(sp_282, inplace=True)
        sp_282 = None
        sp_284 = sp_283 + getitem_94
        getitem_94 = None
        sp_285 = torch.conv2d(
            sp_284,
            l_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_284 = (
            l_self_modules_layer3_modules_16_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_286 = torch.nn.functional.batch_norm(
            sp_285,
            l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_285 = (
            l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_16_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_287 = torch.nn.functional.relu(sp_286, inplace=True)
        sp_286 = None
        out_187 = torch.cat([sp_279, sp_283, sp_287, getitem_95], 1)
        sp_279 = sp_283 = sp_287 = getitem_95 = None
        out_188 = torch.conv2d(
            out_187,
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_187 = (
            l_self_modules_layer3_modules_16_modules_conv3_parameters_weight_
        ) = None
        out_189 = torch.nn.functional.batch_norm(
            out_188,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_188 = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_16_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_16_modules_bn3_parameters_bias_ = None
        out_189 += out_183
        out_190 = out_189
        out_189 = out_183 = None
        out_191 = torch.nn.functional.relu(out_190, inplace=True)
        out_190 = None
        out_192 = torch.conv2d(
            out_191,
            l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_17_modules_conv1_parameters_weight_ = None
        out_193 = torch.nn.functional.batch_norm(
            out_192,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_192 = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn1_parameters_bias_ = None
        out_194 = torch.nn.functional.relu(out_193, inplace=True)
        out_193 = None
        split_24 = torch.functional.split(out_194, 104, 1)
        out_194 = None
        sp_288 = split_24[0]
        getitem_97 = split_24[1]
        getitem_98 = split_24[2]
        getitem_99 = split_24[3]
        split_24 = None
        sp_289 = torch.conv2d(
            sp_288,
            l_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_288 = (
            l_self_modules_layer3_modules_17_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_290 = torch.nn.functional.batch_norm(
            sp_289,
            l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_289 = (
            l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_291 = torch.nn.functional.relu(sp_290, inplace=True)
        sp_290 = None
        sp_292 = sp_291 + getitem_97
        getitem_97 = None
        sp_293 = torch.conv2d(
            sp_292,
            l_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_292 = (
            l_self_modules_layer3_modules_17_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_294 = torch.nn.functional.batch_norm(
            sp_293,
            l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_293 = (
            l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_295 = torch.nn.functional.relu(sp_294, inplace=True)
        sp_294 = None
        sp_296 = sp_295 + getitem_98
        getitem_98 = None
        sp_297 = torch.conv2d(
            sp_296,
            l_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_296 = (
            l_self_modules_layer3_modules_17_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_298 = torch.nn.functional.batch_norm(
            sp_297,
            l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_297 = (
            l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_17_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_299 = torch.nn.functional.relu(sp_298, inplace=True)
        sp_298 = None
        out_195 = torch.cat([sp_291, sp_295, sp_299, getitem_99], 1)
        sp_291 = sp_295 = sp_299 = getitem_99 = None
        out_196 = torch.conv2d(
            out_195,
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_195 = (
            l_self_modules_layer3_modules_17_modules_conv3_parameters_weight_
        ) = None
        out_197 = torch.nn.functional.batch_norm(
            out_196,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_196 = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_17_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_17_modules_bn3_parameters_bias_ = None
        out_197 += out_191
        out_198 = out_197
        out_197 = out_191 = None
        out_199 = torch.nn.functional.relu(out_198, inplace=True)
        out_198 = None
        out_200 = torch.conv2d(
            out_199,
            l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_18_modules_conv1_parameters_weight_ = None
        out_201 = torch.nn.functional.batch_norm(
            out_200,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_200 = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn1_parameters_bias_ = None
        out_202 = torch.nn.functional.relu(out_201, inplace=True)
        out_201 = None
        split_25 = torch.functional.split(out_202, 104, 1)
        out_202 = None
        sp_300 = split_25[0]
        getitem_101 = split_25[1]
        getitem_102 = split_25[2]
        getitem_103 = split_25[3]
        split_25 = None
        sp_301 = torch.conv2d(
            sp_300,
            l_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_300 = (
            l_self_modules_layer3_modules_18_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_302 = torch.nn.functional.batch_norm(
            sp_301,
            l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_301 = (
            l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_303 = torch.nn.functional.relu(sp_302, inplace=True)
        sp_302 = None
        sp_304 = sp_303 + getitem_101
        getitem_101 = None
        sp_305 = torch.conv2d(
            sp_304,
            l_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_304 = (
            l_self_modules_layer3_modules_18_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_306 = torch.nn.functional.batch_norm(
            sp_305,
            l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_305 = (
            l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_307 = torch.nn.functional.relu(sp_306, inplace=True)
        sp_306 = None
        sp_308 = sp_307 + getitem_102
        getitem_102 = None
        sp_309 = torch.conv2d(
            sp_308,
            l_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_308 = (
            l_self_modules_layer3_modules_18_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_310 = torch.nn.functional.batch_norm(
            sp_309,
            l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_309 = (
            l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_18_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_311 = torch.nn.functional.relu(sp_310, inplace=True)
        sp_310 = None
        out_203 = torch.cat([sp_303, sp_307, sp_311, getitem_103], 1)
        sp_303 = sp_307 = sp_311 = getitem_103 = None
        out_204 = torch.conv2d(
            out_203,
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_203 = (
            l_self_modules_layer3_modules_18_modules_conv3_parameters_weight_
        ) = None
        out_205 = torch.nn.functional.batch_norm(
            out_204,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_204 = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_18_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_18_modules_bn3_parameters_bias_ = None
        out_205 += out_199
        out_206 = out_205
        out_205 = out_199 = None
        out_207 = torch.nn.functional.relu(out_206, inplace=True)
        out_206 = None
        out_208 = torch.conv2d(
            out_207,
            l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_19_modules_conv1_parameters_weight_ = None
        out_209 = torch.nn.functional.batch_norm(
            out_208,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_208 = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn1_parameters_bias_ = None
        out_210 = torch.nn.functional.relu(out_209, inplace=True)
        out_209 = None
        split_26 = torch.functional.split(out_210, 104, 1)
        out_210 = None
        sp_312 = split_26[0]
        getitem_105 = split_26[1]
        getitem_106 = split_26[2]
        getitem_107 = split_26[3]
        split_26 = None
        sp_313 = torch.conv2d(
            sp_312,
            l_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_312 = (
            l_self_modules_layer3_modules_19_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_314 = torch.nn.functional.batch_norm(
            sp_313,
            l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_313 = (
            l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_315 = torch.nn.functional.relu(sp_314, inplace=True)
        sp_314 = None
        sp_316 = sp_315 + getitem_105
        getitem_105 = None
        sp_317 = torch.conv2d(
            sp_316,
            l_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_316 = (
            l_self_modules_layer3_modules_19_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_318 = torch.nn.functional.batch_norm(
            sp_317,
            l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_317 = (
            l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_319 = torch.nn.functional.relu(sp_318, inplace=True)
        sp_318 = None
        sp_320 = sp_319 + getitem_106
        getitem_106 = None
        sp_321 = torch.conv2d(
            sp_320,
            l_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_320 = (
            l_self_modules_layer3_modules_19_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_322 = torch.nn.functional.batch_norm(
            sp_321,
            l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_321 = (
            l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_19_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_323 = torch.nn.functional.relu(sp_322, inplace=True)
        sp_322 = None
        out_211 = torch.cat([sp_315, sp_319, sp_323, getitem_107], 1)
        sp_315 = sp_319 = sp_323 = getitem_107 = None
        out_212 = torch.conv2d(
            out_211,
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_211 = (
            l_self_modules_layer3_modules_19_modules_conv3_parameters_weight_
        ) = None
        out_213 = torch.nn.functional.batch_norm(
            out_212,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_212 = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_19_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_19_modules_bn3_parameters_bias_ = None
        out_213 += out_207
        out_214 = out_213
        out_213 = out_207 = None
        out_215 = torch.nn.functional.relu(out_214, inplace=True)
        out_214 = None
        out_216 = torch.conv2d(
            out_215,
            l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_20_modules_conv1_parameters_weight_ = None
        out_217 = torch.nn.functional.batch_norm(
            out_216,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_216 = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn1_parameters_bias_ = None
        out_218 = torch.nn.functional.relu(out_217, inplace=True)
        out_217 = None
        split_27 = torch.functional.split(out_218, 104, 1)
        out_218 = None
        sp_324 = split_27[0]
        getitem_109 = split_27[1]
        getitem_110 = split_27[2]
        getitem_111 = split_27[3]
        split_27 = None
        sp_325 = torch.conv2d(
            sp_324,
            l_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_324 = (
            l_self_modules_layer3_modules_20_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_326 = torch.nn.functional.batch_norm(
            sp_325,
            l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_325 = (
            l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_327 = torch.nn.functional.relu(sp_326, inplace=True)
        sp_326 = None
        sp_328 = sp_327 + getitem_109
        getitem_109 = None
        sp_329 = torch.conv2d(
            sp_328,
            l_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_328 = (
            l_self_modules_layer3_modules_20_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_330 = torch.nn.functional.batch_norm(
            sp_329,
            l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_329 = (
            l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_331 = torch.nn.functional.relu(sp_330, inplace=True)
        sp_330 = None
        sp_332 = sp_331 + getitem_110
        getitem_110 = None
        sp_333 = torch.conv2d(
            sp_332,
            l_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_332 = (
            l_self_modules_layer3_modules_20_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_334 = torch.nn.functional.batch_norm(
            sp_333,
            l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_333 = (
            l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_20_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_335 = torch.nn.functional.relu(sp_334, inplace=True)
        sp_334 = None
        out_219 = torch.cat([sp_327, sp_331, sp_335, getitem_111], 1)
        sp_327 = sp_331 = sp_335 = getitem_111 = None
        out_220 = torch.conv2d(
            out_219,
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_219 = (
            l_self_modules_layer3_modules_20_modules_conv3_parameters_weight_
        ) = None
        out_221 = torch.nn.functional.batch_norm(
            out_220,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_220 = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_20_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_20_modules_bn3_parameters_bias_ = None
        out_221 += out_215
        out_222 = out_221
        out_221 = out_215 = None
        out_223 = torch.nn.functional.relu(out_222, inplace=True)
        out_222 = None
        out_224 = torch.conv2d(
            out_223,
            l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_21_modules_conv1_parameters_weight_ = None
        out_225 = torch.nn.functional.batch_norm(
            out_224,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_224 = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn1_parameters_bias_ = None
        out_226 = torch.nn.functional.relu(out_225, inplace=True)
        out_225 = None
        split_28 = torch.functional.split(out_226, 104, 1)
        out_226 = None
        sp_336 = split_28[0]
        getitem_113 = split_28[1]
        getitem_114 = split_28[2]
        getitem_115 = split_28[3]
        split_28 = None
        sp_337 = torch.conv2d(
            sp_336,
            l_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_336 = (
            l_self_modules_layer3_modules_21_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_338 = torch.nn.functional.batch_norm(
            sp_337,
            l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_337 = (
            l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_339 = torch.nn.functional.relu(sp_338, inplace=True)
        sp_338 = None
        sp_340 = sp_339 + getitem_113
        getitem_113 = None
        sp_341 = torch.conv2d(
            sp_340,
            l_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_340 = (
            l_self_modules_layer3_modules_21_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_342 = torch.nn.functional.batch_norm(
            sp_341,
            l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_341 = (
            l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_343 = torch.nn.functional.relu(sp_342, inplace=True)
        sp_342 = None
        sp_344 = sp_343 + getitem_114
        getitem_114 = None
        sp_345 = torch.conv2d(
            sp_344,
            l_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_344 = (
            l_self_modules_layer3_modules_21_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_346 = torch.nn.functional.batch_norm(
            sp_345,
            l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_345 = (
            l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_21_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_347 = torch.nn.functional.relu(sp_346, inplace=True)
        sp_346 = None
        out_227 = torch.cat([sp_339, sp_343, sp_347, getitem_115], 1)
        sp_339 = sp_343 = sp_347 = getitem_115 = None
        out_228 = torch.conv2d(
            out_227,
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_227 = (
            l_self_modules_layer3_modules_21_modules_conv3_parameters_weight_
        ) = None
        out_229 = torch.nn.functional.batch_norm(
            out_228,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_228 = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_21_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_21_modules_bn3_parameters_bias_ = None
        out_229 += out_223
        out_230 = out_229
        out_229 = out_223 = None
        out_231 = torch.nn.functional.relu(out_230, inplace=True)
        out_230 = None
        out_232 = torch.conv2d(
            out_231,
            l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer3_modules_22_modules_conv1_parameters_weight_ = None
        out_233 = torch.nn.functional.batch_norm(
            out_232,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_232 = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn1_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn1_parameters_bias_ = None
        out_234 = torch.nn.functional.relu(out_233, inplace=True)
        out_233 = None
        split_29 = torch.functional.split(out_234, 104, 1)
        out_234 = None
        sp_348 = split_29[0]
        getitem_117 = split_29[1]
        getitem_118 = split_29[2]
        getitem_119 = split_29[3]
        split_29 = None
        sp_349 = torch.conv2d(
            sp_348,
            l_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_348 = (
            l_self_modules_layer3_modules_22_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_350 = torch.nn.functional.batch_norm(
            sp_349,
            l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_349 = (
            l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_351 = torch.nn.functional.relu(sp_350, inplace=True)
        sp_350 = None
        sp_352 = sp_351 + getitem_117
        getitem_117 = None
        sp_353 = torch.conv2d(
            sp_352,
            l_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_352 = (
            l_self_modules_layer3_modules_22_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_354 = torch.nn.functional.batch_norm(
            sp_353,
            l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_353 = (
            l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_355 = torch.nn.functional.relu(sp_354, inplace=True)
        sp_354 = None
        sp_356 = sp_355 + getitem_118
        getitem_118 = None
        sp_357 = torch.conv2d(
            sp_356,
            l_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_356 = (
            l_self_modules_layer3_modules_22_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_358 = torch.nn.functional.batch_norm(
            sp_357,
            l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_357 = (
            l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_22_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_359 = torch.nn.functional.relu(sp_358, inplace=True)
        sp_358 = None
        out_235 = torch.cat([sp_351, sp_355, sp_359, getitem_119], 1)
        sp_351 = sp_355 = sp_359 = getitem_119 = None
        out_236 = torch.conv2d(
            out_235,
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_235 = (
            l_self_modules_layer3_modules_22_modules_conv3_parameters_weight_
        ) = None
        out_237 = torch.nn.functional.batch_norm(
            out_236,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_,
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_,
            l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_236 = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_22_modules_bn3_parameters_weight_
        ) = l_self_modules_layer3_modules_22_modules_bn3_parameters_bias_ = None
        out_237 += out_231
        out_238 = out_237
        out_237 = out_231 = None
        out_239 = torch.nn.functional.relu(out_238, inplace=True)
        out_238 = None
        out_240 = torch.conv2d(
            out_239,
            l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_0_modules_conv1_parameters_weight_ = None
        out_241 = torch.nn.functional.batch_norm(
            out_240,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_240 = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn1_parameters_bias_ = None
        out_242 = torch.nn.functional.relu(out_241, inplace=True)
        out_241 = None
        split_30 = torch.functional.split(out_242, 208, 1)
        out_242 = None
        sp_360 = split_30[0]
        sp_364 = split_30[1]
        sp_368 = split_30[2]
        getitem_123 = split_30[3]
        split_30 = None
        sp_361 = torch.conv2d(
            sp_360,
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_360 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_362 = torch.nn.functional.batch_norm(
            sp_361,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_361 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_363 = torch.nn.functional.relu(sp_362, inplace=True)
        sp_362 = None
        sp_365 = torch.conv2d(
            sp_364,
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_364 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_366 = torch.nn.functional.batch_norm(
            sp_365,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_365 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_367 = torch.nn.functional.relu(sp_366, inplace=True)
        sp_366 = None
        sp_369 = torch.conv2d(
            sp_368,
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_368 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_370 = torch.nn.functional.batch_norm(
            sp_369,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_369 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_371 = torch.nn.functional.relu(sp_370, inplace=True)
        sp_370 = None
        avg_pool2d_5 = torch._C._nn.avg_pool2d(getitem_123, 3, 2, 1, False, True, None)
        getitem_123 = None
        out_243 = torch.cat([sp_363, sp_367, sp_371, avg_pool2d_5], 1)
        sp_363 = sp_367 = sp_371 = avg_pool2d_5 = None
        out_244 = torch.conv2d(
            out_243,
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_243 = (
            l_self_modules_layer4_modules_0_modules_conv3_parameters_weight_
        ) = None
        out_245 = torch.nn.functional.batch_norm(
            out_244,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_244 = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_0_modules_bn3_parameters_bias_ = None
        input_16 = torch._C._nn.avg_pool2d(out_239, 2, 2, 0, True, False, None)
        out_239 = None
        input_17 = torch.conv2d(
            input_16,
            l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        input_16 = l_self_modules_layer4_modules_0_modules_downsample_modules_1_parameters_weight_ = (None)
        input_18 = torch.nn.functional.batch_norm(
            input_17,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        input_17 = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_mean_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_buffers_running_var_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_weight_ = l_self_modules_layer4_modules_0_modules_downsample_modules_2_parameters_bias_ = (None)
        out_245 += input_18
        out_246 = out_245
        out_245 = input_18 = None
        out_247 = torch.nn.functional.relu(out_246, inplace=True)
        out_246 = None
        out_248 = torch.conv2d(
            out_247,
            l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_1_modules_conv1_parameters_weight_ = None
        out_249 = torch.nn.functional.batch_norm(
            out_248,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_248 = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn1_parameters_bias_ = None
        out_250 = torch.nn.functional.relu(out_249, inplace=True)
        out_249 = None
        split_31 = torch.functional.split(out_250, 208, 1)
        out_250 = None
        sp_372 = split_31[0]
        getitem_125 = split_31[1]
        getitem_126 = split_31[2]
        getitem_127 = split_31[3]
        split_31 = None
        sp_373 = torch.conv2d(
            sp_372,
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_372 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_374 = torch.nn.functional.batch_norm(
            sp_373,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_373 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_375 = torch.nn.functional.relu(sp_374, inplace=True)
        sp_374 = None
        sp_376 = sp_375 + getitem_125
        getitem_125 = None
        sp_377 = torch.conv2d(
            sp_376,
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_376 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_378 = torch.nn.functional.batch_norm(
            sp_377,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_377 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_379 = torch.nn.functional.relu(sp_378, inplace=True)
        sp_378 = None
        sp_380 = sp_379 + getitem_126
        getitem_126 = None
        sp_381 = torch.conv2d(
            sp_380,
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_380 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_382 = torch.nn.functional.batch_norm(
            sp_381,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_381 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_383 = torch.nn.functional.relu(sp_382, inplace=True)
        sp_382 = None
        out_251 = torch.cat([sp_375, sp_379, sp_383, getitem_127], 1)
        sp_375 = sp_379 = sp_383 = getitem_127 = None
        out_252 = torch.conv2d(
            out_251,
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_251 = (
            l_self_modules_layer4_modules_1_modules_conv3_parameters_weight_
        ) = None
        out_253 = torch.nn.functional.batch_norm(
            out_252,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_252 = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_1_modules_bn3_parameters_bias_ = None
        out_253 += out_247
        out_254 = out_253
        out_253 = out_247 = None
        out_255 = torch.nn.functional.relu(out_254, inplace=True)
        out_254 = None
        out_256 = torch.conv2d(
            out_255,
            l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        l_self_modules_layer4_modules_2_modules_conv1_parameters_weight_ = None
        out_257 = torch.nn.functional.batch_norm(
            out_256,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_256 = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn1_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn1_parameters_bias_ = None
        out_258 = torch.nn.functional.relu(out_257, inplace=True)
        out_257 = None
        split_32 = torch.functional.split(out_258, 208, 1)
        out_258 = None
        sp_384 = split_32[0]
        getitem_129 = split_32[1]
        getitem_130 = split_32[2]
        getitem_131 = split_32[3]
        split_32 = None
        sp_385 = torch.conv2d(
            sp_384,
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_384 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_386 = torch.nn.functional.batch_norm(
            sp_385,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_385 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_387 = torch.nn.functional.relu(sp_386, inplace=True)
        sp_386 = None
        sp_388 = sp_387 + getitem_129
        getitem_129 = None
        sp_389 = torch.conv2d(
            sp_388,
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_388 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_390 = torch.nn.functional.batch_norm(
            sp_389,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_389 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_391 = torch.nn.functional.relu(sp_390, inplace=True)
        sp_390 = None
        sp_392 = sp_391 + getitem_130
        getitem_130 = None
        sp_393 = torch.conv2d(
            sp_392,
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_392 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_394 = torch.nn.functional.batch_norm(
            sp_393,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_393 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_395 = torch.nn.functional.relu(sp_394, inplace=True)
        sp_394 = None
        out_259 = torch.cat([sp_387, sp_391, sp_395, getitem_131], 1)
        sp_387 = sp_391 = sp_395 = getitem_131 = None
        out_260 = torch.conv2d(
            out_259,
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_,
            None,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        )
        out_259 = (
            l_self_modules_layer4_modules_2_modules_conv3_parameters_weight_
        ) = None
        out_261 = torch.nn.functional.batch_norm(
            out_260,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        out_260 = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bn3_parameters_weight_
        ) = l_self_modules_layer4_modules_2_modules_bn3_parameters_bias_ = None
        out_261 += out_255
        out_262 = out_261
        out_261 = out_255 = None
        out_263 = torch.nn.functional.relu(out_262, inplace=True)
        out_262 = None
        x_3 = torch.nn.functional.adaptive_avg_pool2d(out_263, 1)
        out_263 = None
        x_4 = x_3.flatten(1, -1)
        x_3 = None
        x_5 = torch._C._nn.linear(
            x_4,
            l_self_modules_fc_parameters_weight_,
            l_self_modules_fc_parameters_bias_,
        )
        x_4 = (
            l_self_modules_fc_parameters_weight_
        ) = l_self_modules_fc_parameters_bias_ = None
        return (x_5,)
