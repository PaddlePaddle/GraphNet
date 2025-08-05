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
        split = torch.functional.split(out_2, 48, 1)
        out_2 = None
        sp = split[0]
        getitem_1 = split[1]
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
        avg_pool2d = torch._C._nn.avg_pool2d(getitem_1, 3, 1, 1, False, True, None)
        getitem_1 = None
        out_3 = torch.cat([sp_3, avg_pool2d], 1)
        sp_3 = avg_pool2d = None
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
        split_1 = torch.functional.split(out_10, 48, 1)
        out_10 = None
        sp_4 = split_1[0]
        getitem_3 = split_1[1]
        split_1 = None
        sp_5 = torch.conv2d(
            sp_4,
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_4 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_6 = torch.nn.functional.batch_norm(
            sp_5,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_5 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_7 = torch.nn.functional.relu(sp_6, inplace=True)
        sp_6 = None
        out_11 = torch.cat([sp_7, getitem_3], 1)
        sp_7 = getitem_3 = None
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
        split_2 = torch.functional.split(out_18, 48, 1)
        out_18 = None
        sp_8 = split_2[0]
        getitem_5 = split_2[1]
        split_2 = None
        sp_9 = torch.conv2d(
            sp_8,
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_8 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_10 = torch.nn.functional.batch_norm(
            sp_9,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_9 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_11 = torch.nn.functional.relu(sp_10, inplace=True)
        sp_10 = None
        out_19 = torch.cat([sp_11, getitem_5], 1)
        sp_11 = getitem_5 = None
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
        split_3 = torch.functional.split(out_26, 96, 1)
        out_26 = None
        sp_12 = split_3[0]
        getitem_7 = split_3[1]
        split_3 = None
        sp_13 = torch.conv2d(
            sp_12,
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_12 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_14 = torch.nn.functional.batch_norm(
            sp_13,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_13 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_15 = torch.nn.functional.relu(sp_14, inplace=True)
        sp_14 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(getitem_7, 3, 2, 1, False, True, None)
        getitem_7 = None
        out_27 = torch.cat([sp_15, avg_pool2d_1], 1)
        sp_15 = avg_pool2d_1 = None
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
        split_4 = torch.functional.split(out_34, 96, 1)
        out_34 = None
        sp_16 = split_4[0]
        getitem_9 = split_4[1]
        split_4 = None
        sp_17 = torch.conv2d(
            sp_16,
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_16 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_18 = torch.nn.functional.batch_norm(
            sp_17,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_17 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_19 = torch.nn.functional.relu(sp_18, inplace=True)
        sp_18 = None
        out_35 = torch.cat([sp_19, getitem_9], 1)
        sp_19 = getitem_9 = None
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
        split_5 = torch.functional.split(out_42, 96, 1)
        out_42 = None
        sp_20 = split_5[0]
        getitem_11 = split_5[1]
        split_5 = None
        sp_21 = torch.conv2d(
            sp_20,
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_20 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_22 = torch.nn.functional.batch_norm(
            sp_21,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_21 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_23 = torch.nn.functional.relu(sp_22, inplace=True)
        sp_22 = None
        out_43 = torch.cat([sp_23, getitem_11], 1)
        sp_23 = getitem_11 = None
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
        split_6 = torch.functional.split(out_50, 96, 1)
        out_50 = None
        sp_24 = split_6[0]
        getitem_13 = split_6[1]
        split_6 = None
        sp_25 = torch.conv2d(
            sp_24,
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_24 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_26 = torch.nn.functional.batch_norm(
            sp_25,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_25 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_27 = torch.nn.functional.relu(sp_26, inplace=True)
        sp_26 = None
        out_51 = torch.cat([sp_27, getitem_13], 1)
        sp_27 = getitem_13 = None
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
        split_7 = torch.functional.split(out_58, 192, 1)
        out_58 = None
        sp_28 = split_7[0]
        getitem_15 = split_7[1]
        split_7 = None
        sp_29 = torch.conv2d(
            sp_28,
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_28 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_30 = torch.nn.functional.batch_norm(
            sp_29,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_29 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_31 = torch.nn.functional.relu(sp_30, inplace=True)
        sp_30 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(getitem_15, 3, 2, 1, False, True, None)
        getitem_15 = None
        out_59 = torch.cat([sp_31, avg_pool2d_2], 1)
        sp_31 = avg_pool2d_2 = None
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
        split_8 = torch.functional.split(out_66, 192, 1)
        out_66 = None
        sp_32 = split_8[0]
        getitem_17 = split_8[1]
        split_8 = None
        sp_33 = torch.conv2d(
            sp_32,
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_32 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_34 = torch.nn.functional.batch_norm(
            sp_33,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_33 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_35 = torch.nn.functional.relu(sp_34, inplace=True)
        sp_34 = None
        out_67 = torch.cat([sp_35, getitem_17], 1)
        sp_35 = getitem_17 = None
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
        split_9 = torch.functional.split(out_74, 192, 1)
        out_74 = None
        sp_36 = split_9[0]
        getitem_19 = split_9[1]
        split_9 = None
        sp_37 = torch.conv2d(
            sp_36,
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_36 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_38 = torch.nn.functional.batch_norm(
            sp_37,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_37 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_39 = torch.nn.functional.relu(sp_38, inplace=True)
        sp_38 = None
        out_75 = torch.cat([sp_39, getitem_19], 1)
        sp_39 = getitem_19 = None
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
        split_10 = torch.functional.split(out_82, 192, 1)
        out_82 = None
        sp_40 = split_10[0]
        getitem_21 = split_10[1]
        split_10 = None
        sp_41 = torch.conv2d(
            sp_40,
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_40 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_42 = torch.nn.functional.batch_norm(
            sp_41,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_41 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_43 = torch.nn.functional.relu(sp_42, inplace=True)
        sp_42 = None
        out_83 = torch.cat([sp_43, getitem_21], 1)
        sp_43 = getitem_21 = None
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
        split_11 = torch.functional.split(out_90, 192, 1)
        out_90 = None
        sp_44 = split_11[0]
        getitem_23 = split_11[1]
        split_11 = None
        sp_45 = torch.conv2d(
            sp_44,
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_44 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_46 = torch.nn.functional.batch_norm(
            sp_45,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_45 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_47 = torch.nn.functional.relu(sp_46, inplace=True)
        sp_46 = None
        out_91 = torch.cat([sp_47, getitem_23], 1)
        sp_47 = getitem_23 = None
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
        split_12 = torch.functional.split(out_98, 192, 1)
        out_98 = None
        sp_48 = split_12[0]
        getitem_25 = split_12[1]
        split_12 = None
        sp_49 = torch.conv2d(
            sp_48,
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_48 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_50 = torch.nn.functional.batch_norm(
            sp_49,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_49 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_51 = torch.nn.functional.relu(sp_50, inplace=True)
        sp_50 = None
        out_99 = torch.cat([sp_51, getitem_25], 1)
        sp_51 = getitem_25 = None
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
        split_13 = torch.functional.split(out_106, 384, 1)
        out_106 = None
        sp_52 = split_13[0]
        getitem_27 = split_13[1]
        split_13 = None
        sp_53 = torch.conv2d(
            sp_52,
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_52 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_54 = torch.nn.functional.batch_norm(
            sp_53,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_53 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_55 = torch.nn.functional.relu(sp_54, inplace=True)
        sp_54 = None
        avg_pool2d_3 = torch._C._nn.avg_pool2d(getitem_27, 3, 2, 1, False, True, None)
        getitem_27 = None
        out_107 = torch.cat([sp_55, avg_pool2d_3], 1)
        sp_55 = avg_pool2d_3 = None
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
        split_14 = torch.functional.split(out_114, 384, 1)
        out_114 = None
        sp_56 = split_14[0]
        getitem_29 = split_14[1]
        split_14 = None
        sp_57 = torch.conv2d(
            sp_56,
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_56 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_58 = torch.nn.functional.batch_norm(
            sp_57,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_57 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_59 = torch.nn.functional.relu(sp_58, inplace=True)
        sp_58 = None
        out_115 = torch.cat([sp_59, getitem_29], 1)
        sp_59 = getitem_29 = None
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
        split_15 = torch.functional.split(out_122, 384, 1)
        out_122 = None
        sp_60 = split_15[0]
        getitem_31 = split_15[1]
        split_15 = None
        sp_61 = torch.conv2d(
            sp_60,
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_60 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_62 = torch.nn.functional.batch_norm(
            sp_61,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_61 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_63 = torch.nn.functional.relu(sp_62, inplace=True)
        sp_62 = None
        out_123 = torch.cat([sp_63, getitem_31], 1)
        sp_63 = getitem_31 = None
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
