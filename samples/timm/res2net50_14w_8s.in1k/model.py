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
        L_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        L_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_: torch.Tensor,
        L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_: torch.nn.parameter.Parameter,
        L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_: torch.nn.parameter.Parameter,
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
        l_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_
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
        l_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_
        )
        l_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_
        )
        l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_ = (
            L_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_
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
        split = torch.functional.split(out_2, 14, 1)
        out_2 = None
        sp = split[0]
        sp_4 = split[1]
        sp_8 = split[2]
        sp_12 = split[3]
        sp_16 = split[4]
        sp_20 = split[5]
        sp_24 = split[6]
        getitem_7 = split[7]
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
        sp_13 = torch.conv2d(
            sp_12,
            l_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_12 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_14 = torch.nn.functional.batch_norm(
            sp_13,
            l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_13 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_15 = torch.nn.functional.relu(sp_14, inplace=True)
        sp_14 = None
        sp_17 = torch.conv2d(
            sp_16,
            l_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_16 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_18 = torch.nn.functional.batch_norm(
            sp_17,
            l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_17 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_19 = torch.nn.functional.relu(sp_18, inplace=True)
        sp_18 = None
        sp_21 = torch.conv2d(
            sp_20,
            l_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_20 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_22 = torch.nn.functional.batch_norm(
            sp_21,
            l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_21 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_23 = torch.nn.functional.relu(sp_22, inplace=True)
        sp_22 = None
        sp_25 = torch.conv2d(
            sp_24,
            l_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_24 = (
            l_self_modules_layer1_modules_0_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_26 = torch.nn.functional.batch_norm(
            sp_25,
            l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_25 = (
            l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_0_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_27 = torch.nn.functional.relu(sp_26, inplace=True)
        sp_26 = None
        avg_pool2d = torch._C._nn.avg_pool2d(getitem_7, 3, 1, 1, False, True, None)
        getitem_7 = None
        out_3 = torch.cat(
            [sp_3, sp_7, sp_11, sp_15, sp_19, sp_23, sp_27, avg_pool2d], 1
        )
        sp_3 = sp_7 = sp_11 = sp_15 = sp_19 = sp_23 = sp_27 = avg_pool2d = None
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
        split_1 = torch.functional.split(out_10, 14, 1)
        out_10 = None
        sp_28 = split_1[0]
        getitem_9 = split_1[1]
        getitem_10 = split_1[2]
        getitem_11 = split_1[3]
        getitem_12 = split_1[4]
        getitem_13 = split_1[5]
        getitem_14 = split_1[6]
        getitem_15 = split_1[7]
        split_1 = None
        sp_29 = torch.conv2d(
            sp_28,
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_28 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_30 = torch.nn.functional.batch_norm(
            sp_29,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_29 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_31 = torch.nn.functional.relu(sp_30, inplace=True)
        sp_30 = None
        sp_32 = sp_31 + getitem_9
        getitem_9 = None
        sp_33 = torch.conv2d(
            sp_32,
            l_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_32 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_34 = torch.nn.functional.batch_norm(
            sp_33,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_33 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_35 = torch.nn.functional.relu(sp_34, inplace=True)
        sp_34 = None
        sp_36 = sp_35 + getitem_10
        getitem_10 = None
        sp_37 = torch.conv2d(
            sp_36,
            l_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_36 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_38 = torch.nn.functional.batch_norm(
            sp_37,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_37 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_39 = torch.nn.functional.relu(sp_38, inplace=True)
        sp_38 = None
        sp_40 = sp_39 + getitem_11
        getitem_11 = None
        sp_41 = torch.conv2d(
            sp_40,
            l_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_40 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_42 = torch.nn.functional.batch_norm(
            sp_41,
            l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_41 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_43 = torch.nn.functional.relu(sp_42, inplace=True)
        sp_42 = None
        sp_44 = sp_43 + getitem_12
        getitem_12 = None
        sp_45 = torch.conv2d(
            sp_44,
            l_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_44 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_46 = torch.nn.functional.batch_norm(
            sp_45,
            l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_45 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_47 = torch.nn.functional.relu(sp_46, inplace=True)
        sp_46 = None
        sp_48 = sp_47 + getitem_13
        getitem_13 = None
        sp_49 = torch.conv2d(
            sp_48,
            l_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_48 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_50 = torch.nn.functional.batch_norm(
            sp_49,
            l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_49 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_51 = torch.nn.functional.relu(sp_50, inplace=True)
        sp_50 = None
        sp_52 = sp_51 + getitem_14
        getitem_14 = None
        sp_53 = torch.conv2d(
            sp_52,
            l_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_52 = (
            l_self_modules_layer1_modules_1_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_54 = torch.nn.functional.batch_norm(
            sp_53,
            l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_53 = (
            l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_1_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_55 = torch.nn.functional.relu(sp_54, inplace=True)
        sp_54 = None
        out_11 = torch.cat(
            [sp_31, sp_35, sp_39, sp_43, sp_47, sp_51, sp_55, getitem_15], 1
        )
        sp_31 = sp_35 = sp_39 = sp_43 = sp_47 = sp_51 = sp_55 = getitem_15 = None
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
        split_2 = torch.functional.split(out_18, 14, 1)
        out_18 = None
        sp_56 = split_2[0]
        getitem_17 = split_2[1]
        getitem_18 = split_2[2]
        getitem_19 = split_2[3]
        getitem_20 = split_2[4]
        getitem_21 = split_2[5]
        getitem_22 = split_2[6]
        getitem_23 = split_2[7]
        split_2 = None
        sp_57 = torch.conv2d(
            sp_56,
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_56 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_58 = torch.nn.functional.batch_norm(
            sp_57,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_57 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_59 = torch.nn.functional.relu(sp_58, inplace=True)
        sp_58 = None
        sp_60 = sp_59 + getitem_17
        getitem_17 = None
        sp_61 = torch.conv2d(
            sp_60,
            l_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_60 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_62 = torch.nn.functional.batch_norm(
            sp_61,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_61 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_63 = torch.nn.functional.relu(sp_62, inplace=True)
        sp_62 = None
        sp_64 = sp_63 + getitem_18
        getitem_18 = None
        sp_65 = torch.conv2d(
            sp_64,
            l_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_64 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_66 = torch.nn.functional.batch_norm(
            sp_65,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_65 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_67 = torch.nn.functional.relu(sp_66, inplace=True)
        sp_66 = None
        sp_68 = sp_67 + getitem_19
        getitem_19 = None
        sp_69 = torch.conv2d(
            sp_68,
            l_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_68 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_70 = torch.nn.functional.batch_norm(
            sp_69,
            l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_69 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_71 = torch.nn.functional.relu(sp_70, inplace=True)
        sp_70 = None
        sp_72 = sp_71 + getitem_20
        getitem_20 = None
        sp_73 = torch.conv2d(
            sp_72,
            l_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_72 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_74 = torch.nn.functional.batch_norm(
            sp_73,
            l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_73 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_75 = torch.nn.functional.relu(sp_74, inplace=True)
        sp_74 = None
        sp_76 = sp_75 + getitem_21
        getitem_21 = None
        sp_77 = torch.conv2d(
            sp_76,
            l_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_76 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_78 = torch.nn.functional.batch_norm(
            sp_77,
            l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_77 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_79 = torch.nn.functional.relu(sp_78, inplace=True)
        sp_78 = None
        sp_80 = sp_79 + getitem_22
        getitem_22 = None
        sp_81 = torch.conv2d(
            sp_80,
            l_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_80 = (
            l_self_modules_layer1_modules_2_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_82 = torch.nn.functional.batch_norm(
            sp_81,
            l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_81 = (
            l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer1_modules_2_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_83 = torch.nn.functional.relu(sp_82, inplace=True)
        sp_82 = None
        out_19 = torch.cat(
            [sp_59, sp_63, sp_67, sp_71, sp_75, sp_79, sp_83, getitem_23], 1
        )
        sp_59 = sp_63 = sp_67 = sp_71 = sp_75 = sp_79 = sp_83 = getitem_23 = None
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
        split_3 = torch.functional.split(out_26, 28, 1)
        out_26 = None
        sp_84 = split_3[0]
        sp_88 = split_3[1]
        sp_92 = split_3[2]
        sp_96 = split_3[3]
        sp_100 = split_3[4]
        sp_104 = split_3[5]
        sp_108 = split_3[6]
        getitem_31 = split_3[7]
        split_3 = None
        sp_85 = torch.conv2d(
            sp_84,
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_84 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_86 = torch.nn.functional.batch_norm(
            sp_85,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_85 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_87 = torch.nn.functional.relu(sp_86, inplace=True)
        sp_86 = None
        sp_89 = torch.conv2d(
            sp_88,
            l_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_88 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_90 = torch.nn.functional.batch_norm(
            sp_89,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_89 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_91 = torch.nn.functional.relu(sp_90, inplace=True)
        sp_90 = None
        sp_93 = torch.conv2d(
            sp_92,
            l_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_92 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_94 = torch.nn.functional.batch_norm(
            sp_93,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_93 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_95 = torch.nn.functional.relu(sp_94, inplace=True)
        sp_94 = None
        sp_97 = torch.conv2d(
            sp_96,
            l_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_96 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_98 = torch.nn.functional.batch_norm(
            sp_97,
            l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_97 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_99 = torch.nn.functional.relu(sp_98, inplace=True)
        sp_98 = None
        sp_101 = torch.conv2d(
            sp_100,
            l_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_100 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_102 = torch.nn.functional.batch_norm(
            sp_101,
            l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_101 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_103 = torch.nn.functional.relu(sp_102, inplace=True)
        sp_102 = None
        sp_105 = torch.conv2d(
            sp_104,
            l_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_104 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_106 = torch.nn.functional.batch_norm(
            sp_105,
            l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_105 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_107 = torch.nn.functional.relu(sp_106, inplace=True)
        sp_106 = None
        sp_109 = torch.conv2d(
            sp_108,
            l_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_108 = (
            l_self_modules_layer2_modules_0_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_110 = torch.nn.functional.batch_norm(
            sp_109,
            l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_109 = (
            l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_0_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_111 = torch.nn.functional.relu(sp_110, inplace=True)
        sp_110 = None
        avg_pool2d_1 = torch._C._nn.avg_pool2d(getitem_31, 3, 2, 1, False, True, None)
        getitem_31 = None
        out_27 = torch.cat(
            [sp_87, sp_91, sp_95, sp_99, sp_103, sp_107, sp_111, avg_pool2d_1], 1
        )
        sp_87 = sp_91 = sp_95 = sp_99 = sp_103 = sp_107 = sp_111 = avg_pool2d_1 = None
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
        split_4 = torch.functional.split(out_34, 28, 1)
        out_34 = None
        sp_112 = split_4[0]
        getitem_33 = split_4[1]
        getitem_34 = split_4[2]
        getitem_35 = split_4[3]
        getitem_36 = split_4[4]
        getitem_37 = split_4[5]
        getitem_38 = split_4[6]
        getitem_39 = split_4[7]
        split_4 = None
        sp_113 = torch.conv2d(
            sp_112,
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_112 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_114 = torch.nn.functional.batch_norm(
            sp_113,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_113 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_115 = torch.nn.functional.relu(sp_114, inplace=True)
        sp_114 = None
        sp_116 = sp_115 + getitem_33
        getitem_33 = None
        sp_117 = torch.conv2d(
            sp_116,
            l_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_116 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_118 = torch.nn.functional.batch_norm(
            sp_117,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_117 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_119 = torch.nn.functional.relu(sp_118, inplace=True)
        sp_118 = None
        sp_120 = sp_119 + getitem_34
        getitem_34 = None
        sp_121 = torch.conv2d(
            sp_120,
            l_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_120 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_122 = torch.nn.functional.batch_norm(
            sp_121,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_121 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_123 = torch.nn.functional.relu(sp_122, inplace=True)
        sp_122 = None
        sp_124 = sp_123 + getitem_35
        getitem_35 = None
        sp_125 = torch.conv2d(
            sp_124,
            l_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_124 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_126 = torch.nn.functional.batch_norm(
            sp_125,
            l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_125 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_127 = torch.nn.functional.relu(sp_126, inplace=True)
        sp_126 = None
        sp_128 = sp_127 + getitem_36
        getitem_36 = None
        sp_129 = torch.conv2d(
            sp_128,
            l_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_128 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_130 = torch.nn.functional.batch_norm(
            sp_129,
            l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_129 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_131 = torch.nn.functional.relu(sp_130, inplace=True)
        sp_130 = None
        sp_132 = sp_131 + getitem_37
        getitem_37 = None
        sp_133 = torch.conv2d(
            sp_132,
            l_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_132 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_134 = torch.nn.functional.batch_norm(
            sp_133,
            l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_133 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_135 = torch.nn.functional.relu(sp_134, inplace=True)
        sp_134 = None
        sp_136 = sp_135 + getitem_38
        getitem_38 = None
        sp_137 = torch.conv2d(
            sp_136,
            l_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_136 = (
            l_self_modules_layer2_modules_1_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_138 = torch.nn.functional.batch_norm(
            sp_137,
            l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_137 = (
            l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_1_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_139 = torch.nn.functional.relu(sp_138, inplace=True)
        sp_138 = None
        out_35 = torch.cat(
            [sp_115, sp_119, sp_123, sp_127, sp_131, sp_135, sp_139, getitem_39], 1
        )
        sp_115 = sp_119 = sp_123 = sp_127 = sp_131 = sp_135 = sp_139 = getitem_39 = None
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
        split_5 = torch.functional.split(out_42, 28, 1)
        out_42 = None
        sp_140 = split_5[0]
        getitem_41 = split_5[1]
        getitem_42 = split_5[2]
        getitem_43 = split_5[3]
        getitem_44 = split_5[4]
        getitem_45 = split_5[5]
        getitem_46 = split_5[6]
        getitem_47 = split_5[7]
        split_5 = None
        sp_141 = torch.conv2d(
            sp_140,
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_140 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_142 = torch.nn.functional.batch_norm(
            sp_141,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_141 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_143 = torch.nn.functional.relu(sp_142, inplace=True)
        sp_142 = None
        sp_144 = sp_143 + getitem_41
        getitem_41 = None
        sp_145 = torch.conv2d(
            sp_144,
            l_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_144 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_146 = torch.nn.functional.batch_norm(
            sp_145,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_145 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_147 = torch.nn.functional.relu(sp_146, inplace=True)
        sp_146 = None
        sp_148 = sp_147 + getitem_42
        getitem_42 = None
        sp_149 = torch.conv2d(
            sp_148,
            l_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_148 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_150 = torch.nn.functional.batch_norm(
            sp_149,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_149 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_151 = torch.nn.functional.relu(sp_150, inplace=True)
        sp_150 = None
        sp_152 = sp_151 + getitem_43
        getitem_43 = None
        sp_153 = torch.conv2d(
            sp_152,
            l_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_152 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_154 = torch.nn.functional.batch_norm(
            sp_153,
            l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_153 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_155 = torch.nn.functional.relu(sp_154, inplace=True)
        sp_154 = None
        sp_156 = sp_155 + getitem_44
        getitem_44 = None
        sp_157 = torch.conv2d(
            sp_156,
            l_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_156 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_158 = torch.nn.functional.batch_norm(
            sp_157,
            l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_157 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_159 = torch.nn.functional.relu(sp_158, inplace=True)
        sp_158 = None
        sp_160 = sp_159 + getitem_45
        getitem_45 = None
        sp_161 = torch.conv2d(
            sp_160,
            l_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_160 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_162 = torch.nn.functional.batch_norm(
            sp_161,
            l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_161 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_163 = torch.nn.functional.relu(sp_162, inplace=True)
        sp_162 = None
        sp_164 = sp_163 + getitem_46
        getitem_46 = None
        sp_165 = torch.conv2d(
            sp_164,
            l_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_164 = (
            l_self_modules_layer2_modules_2_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_166 = torch.nn.functional.batch_norm(
            sp_165,
            l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_165 = (
            l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_2_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_167 = torch.nn.functional.relu(sp_166, inplace=True)
        sp_166 = None
        out_43 = torch.cat(
            [sp_143, sp_147, sp_151, sp_155, sp_159, sp_163, sp_167, getitem_47], 1
        )
        sp_143 = sp_147 = sp_151 = sp_155 = sp_159 = sp_163 = sp_167 = getitem_47 = None
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
        split_6 = torch.functional.split(out_50, 28, 1)
        out_50 = None
        sp_168 = split_6[0]
        getitem_49 = split_6[1]
        getitem_50 = split_6[2]
        getitem_51 = split_6[3]
        getitem_52 = split_6[4]
        getitem_53 = split_6[5]
        getitem_54 = split_6[6]
        getitem_55 = split_6[7]
        split_6 = None
        sp_169 = torch.conv2d(
            sp_168,
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_168 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_170 = torch.nn.functional.batch_norm(
            sp_169,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_169 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_171 = torch.nn.functional.relu(sp_170, inplace=True)
        sp_170 = None
        sp_172 = sp_171 + getitem_49
        getitem_49 = None
        sp_173 = torch.conv2d(
            sp_172,
            l_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_172 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_174 = torch.nn.functional.batch_norm(
            sp_173,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_173 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_175 = torch.nn.functional.relu(sp_174, inplace=True)
        sp_174 = None
        sp_176 = sp_175 + getitem_50
        getitem_50 = None
        sp_177 = torch.conv2d(
            sp_176,
            l_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_176 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_178 = torch.nn.functional.batch_norm(
            sp_177,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_177 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_179 = torch.nn.functional.relu(sp_178, inplace=True)
        sp_178 = None
        sp_180 = sp_179 + getitem_51
        getitem_51 = None
        sp_181 = torch.conv2d(
            sp_180,
            l_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_180 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_182 = torch.nn.functional.batch_norm(
            sp_181,
            l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_181 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_183 = torch.nn.functional.relu(sp_182, inplace=True)
        sp_182 = None
        sp_184 = sp_183 + getitem_52
        getitem_52 = None
        sp_185 = torch.conv2d(
            sp_184,
            l_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_184 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_186 = torch.nn.functional.batch_norm(
            sp_185,
            l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_185 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_187 = torch.nn.functional.relu(sp_186, inplace=True)
        sp_186 = None
        sp_188 = sp_187 + getitem_53
        getitem_53 = None
        sp_189 = torch.conv2d(
            sp_188,
            l_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_188 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_190 = torch.nn.functional.batch_norm(
            sp_189,
            l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_189 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_191 = torch.nn.functional.relu(sp_190, inplace=True)
        sp_190 = None
        sp_192 = sp_191 + getitem_54
        getitem_54 = None
        sp_193 = torch.conv2d(
            sp_192,
            l_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_192 = (
            l_self_modules_layer2_modules_3_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_194 = torch.nn.functional.batch_norm(
            sp_193,
            l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_193 = (
            l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer2_modules_3_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_195 = torch.nn.functional.relu(sp_194, inplace=True)
        sp_194 = None
        out_51 = torch.cat(
            [sp_171, sp_175, sp_179, sp_183, sp_187, sp_191, sp_195, getitem_55], 1
        )
        sp_171 = sp_175 = sp_179 = sp_183 = sp_187 = sp_191 = sp_195 = getitem_55 = None
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
        split_7 = torch.functional.split(out_58, 56, 1)
        out_58 = None
        sp_196 = split_7[0]
        sp_200 = split_7[1]
        sp_204 = split_7[2]
        sp_208 = split_7[3]
        sp_212 = split_7[4]
        sp_216 = split_7[5]
        sp_220 = split_7[6]
        getitem_63 = split_7[7]
        split_7 = None
        sp_197 = torch.conv2d(
            sp_196,
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_196 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_198 = torch.nn.functional.batch_norm(
            sp_197,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_197 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_199 = torch.nn.functional.relu(sp_198, inplace=True)
        sp_198 = None
        sp_201 = torch.conv2d(
            sp_200,
            l_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_200 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_202 = torch.nn.functional.batch_norm(
            sp_201,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_201 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_203 = torch.nn.functional.relu(sp_202, inplace=True)
        sp_202 = None
        sp_205 = torch.conv2d(
            sp_204,
            l_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_204 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_206 = torch.nn.functional.batch_norm(
            sp_205,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_205 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_207 = torch.nn.functional.relu(sp_206, inplace=True)
        sp_206 = None
        sp_209 = torch.conv2d(
            sp_208,
            l_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_208 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_210 = torch.nn.functional.batch_norm(
            sp_209,
            l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_209 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_211 = torch.nn.functional.relu(sp_210, inplace=True)
        sp_210 = None
        sp_213 = torch.conv2d(
            sp_212,
            l_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_212 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_214 = torch.nn.functional.batch_norm(
            sp_213,
            l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_213 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_215 = torch.nn.functional.relu(sp_214, inplace=True)
        sp_214 = None
        sp_217 = torch.conv2d(
            sp_216,
            l_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_216 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_218 = torch.nn.functional.batch_norm(
            sp_217,
            l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_217 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_219 = torch.nn.functional.relu(sp_218, inplace=True)
        sp_218 = None
        sp_221 = torch.conv2d(
            sp_220,
            l_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_220 = (
            l_self_modules_layer3_modules_0_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_222 = torch.nn.functional.batch_norm(
            sp_221,
            l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_221 = (
            l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_0_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_223 = torch.nn.functional.relu(sp_222, inplace=True)
        sp_222 = None
        avg_pool2d_2 = torch._C._nn.avg_pool2d(getitem_63, 3, 2, 1, False, True, None)
        getitem_63 = None
        out_59 = torch.cat(
            [sp_199, sp_203, sp_207, sp_211, sp_215, sp_219, sp_223, avg_pool2d_2], 1
        )
        sp_199 = (
            sp_203
        ) = sp_207 = sp_211 = sp_215 = sp_219 = sp_223 = avg_pool2d_2 = None
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
        split_8 = torch.functional.split(out_66, 56, 1)
        out_66 = None
        sp_224 = split_8[0]
        getitem_65 = split_8[1]
        getitem_66 = split_8[2]
        getitem_67 = split_8[3]
        getitem_68 = split_8[4]
        getitem_69 = split_8[5]
        getitem_70 = split_8[6]
        getitem_71 = split_8[7]
        split_8 = None
        sp_225 = torch.conv2d(
            sp_224,
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_224 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_226 = torch.nn.functional.batch_norm(
            sp_225,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_225 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_227 = torch.nn.functional.relu(sp_226, inplace=True)
        sp_226 = None
        sp_228 = sp_227 + getitem_65
        getitem_65 = None
        sp_229 = torch.conv2d(
            sp_228,
            l_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_228 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_230 = torch.nn.functional.batch_norm(
            sp_229,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_229 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_231 = torch.nn.functional.relu(sp_230, inplace=True)
        sp_230 = None
        sp_232 = sp_231 + getitem_66
        getitem_66 = None
        sp_233 = torch.conv2d(
            sp_232,
            l_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_232 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_234 = torch.nn.functional.batch_norm(
            sp_233,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_233 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_235 = torch.nn.functional.relu(sp_234, inplace=True)
        sp_234 = None
        sp_236 = sp_235 + getitem_67
        getitem_67 = None
        sp_237 = torch.conv2d(
            sp_236,
            l_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_236 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_238 = torch.nn.functional.batch_norm(
            sp_237,
            l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_237 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_239 = torch.nn.functional.relu(sp_238, inplace=True)
        sp_238 = None
        sp_240 = sp_239 + getitem_68
        getitem_68 = None
        sp_241 = torch.conv2d(
            sp_240,
            l_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_240 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_242 = torch.nn.functional.batch_norm(
            sp_241,
            l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_241 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_243 = torch.nn.functional.relu(sp_242, inplace=True)
        sp_242 = None
        sp_244 = sp_243 + getitem_69
        getitem_69 = None
        sp_245 = torch.conv2d(
            sp_244,
            l_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_244 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_246 = torch.nn.functional.batch_norm(
            sp_245,
            l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_245 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_247 = torch.nn.functional.relu(sp_246, inplace=True)
        sp_246 = None
        sp_248 = sp_247 + getitem_70
        getitem_70 = None
        sp_249 = torch.conv2d(
            sp_248,
            l_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_248 = (
            l_self_modules_layer3_modules_1_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_250 = torch.nn.functional.batch_norm(
            sp_249,
            l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_249 = (
            l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_1_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_251 = torch.nn.functional.relu(sp_250, inplace=True)
        sp_250 = None
        out_67 = torch.cat(
            [sp_227, sp_231, sp_235, sp_239, sp_243, sp_247, sp_251, getitem_71], 1
        )
        sp_227 = sp_231 = sp_235 = sp_239 = sp_243 = sp_247 = sp_251 = getitem_71 = None
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
        split_9 = torch.functional.split(out_74, 56, 1)
        out_74 = None
        sp_252 = split_9[0]
        getitem_73 = split_9[1]
        getitem_74 = split_9[2]
        getitem_75 = split_9[3]
        getitem_76 = split_9[4]
        getitem_77 = split_9[5]
        getitem_78 = split_9[6]
        getitem_79 = split_9[7]
        split_9 = None
        sp_253 = torch.conv2d(
            sp_252,
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_252 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_254 = torch.nn.functional.batch_norm(
            sp_253,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_253 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_255 = torch.nn.functional.relu(sp_254, inplace=True)
        sp_254 = None
        sp_256 = sp_255 + getitem_73
        getitem_73 = None
        sp_257 = torch.conv2d(
            sp_256,
            l_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_256 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_258 = torch.nn.functional.batch_norm(
            sp_257,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_257 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_259 = torch.nn.functional.relu(sp_258, inplace=True)
        sp_258 = None
        sp_260 = sp_259 + getitem_74
        getitem_74 = None
        sp_261 = torch.conv2d(
            sp_260,
            l_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_260 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_262 = torch.nn.functional.batch_norm(
            sp_261,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_261 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_263 = torch.nn.functional.relu(sp_262, inplace=True)
        sp_262 = None
        sp_264 = sp_263 + getitem_75
        getitem_75 = None
        sp_265 = torch.conv2d(
            sp_264,
            l_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_264 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_266 = torch.nn.functional.batch_norm(
            sp_265,
            l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_265 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_267 = torch.nn.functional.relu(sp_266, inplace=True)
        sp_266 = None
        sp_268 = sp_267 + getitem_76
        getitem_76 = None
        sp_269 = torch.conv2d(
            sp_268,
            l_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_268 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_270 = torch.nn.functional.batch_norm(
            sp_269,
            l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_269 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_271 = torch.nn.functional.relu(sp_270, inplace=True)
        sp_270 = None
        sp_272 = sp_271 + getitem_77
        getitem_77 = None
        sp_273 = torch.conv2d(
            sp_272,
            l_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_272 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_274 = torch.nn.functional.batch_norm(
            sp_273,
            l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_273 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_275 = torch.nn.functional.relu(sp_274, inplace=True)
        sp_274 = None
        sp_276 = sp_275 + getitem_78
        getitem_78 = None
        sp_277 = torch.conv2d(
            sp_276,
            l_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_276 = (
            l_self_modules_layer3_modules_2_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_278 = torch.nn.functional.batch_norm(
            sp_277,
            l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_277 = (
            l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_2_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_279 = torch.nn.functional.relu(sp_278, inplace=True)
        sp_278 = None
        out_75 = torch.cat(
            [sp_255, sp_259, sp_263, sp_267, sp_271, sp_275, sp_279, getitem_79], 1
        )
        sp_255 = sp_259 = sp_263 = sp_267 = sp_271 = sp_275 = sp_279 = getitem_79 = None
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
        split_10 = torch.functional.split(out_82, 56, 1)
        out_82 = None
        sp_280 = split_10[0]
        getitem_81 = split_10[1]
        getitem_82 = split_10[2]
        getitem_83 = split_10[3]
        getitem_84 = split_10[4]
        getitem_85 = split_10[5]
        getitem_86 = split_10[6]
        getitem_87 = split_10[7]
        split_10 = None
        sp_281 = torch.conv2d(
            sp_280,
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_280 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_282 = torch.nn.functional.batch_norm(
            sp_281,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_281 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_283 = torch.nn.functional.relu(sp_282, inplace=True)
        sp_282 = None
        sp_284 = sp_283 + getitem_81
        getitem_81 = None
        sp_285 = torch.conv2d(
            sp_284,
            l_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_284 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_286 = torch.nn.functional.batch_norm(
            sp_285,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_285 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_287 = torch.nn.functional.relu(sp_286, inplace=True)
        sp_286 = None
        sp_288 = sp_287 + getitem_82
        getitem_82 = None
        sp_289 = torch.conv2d(
            sp_288,
            l_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_288 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_290 = torch.nn.functional.batch_norm(
            sp_289,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_289 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_291 = torch.nn.functional.relu(sp_290, inplace=True)
        sp_290 = None
        sp_292 = sp_291 + getitem_83
        getitem_83 = None
        sp_293 = torch.conv2d(
            sp_292,
            l_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_292 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_294 = torch.nn.functional.batch_norm(
            sp_293,
            l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_293 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_295 = torch.nn.functional.relu(sp_294, inplace=True)
        sp_294 = None
        sp_296 = sp_295 + getitem_84
        getitem_84 = None
        sp_297 = torch.conv2d(
            sp_296,
            l_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_296 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_298 = torch.nn.functional.batch_norm(
            sp_297,
            l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_297 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_299 = torch.nn.functional.relu(sp_298, inplace=True)
        sp_298 = None
        sp_300 = sp_299 + getitem_85
        getitem_85 = None
        sp_301 = torch.conv2d(
            sp_300,
            l_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_300 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_302 = torch.nn.functional.batch_norm(
            sp_301,
            l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_301 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_303 = torch.nn.functional.relu(sp_302, inplace=True)
        sp_302 = None
        sp_304 = sp_303 + getitem_86
        getitem_86 = None
        sp_305 = torch.conv2d(
            sp_304,
            l_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_304 = (
            l_self_modules_layer3_modules_3_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_306 = torch.nn.functional.batch_norm(
            sp_305,
            l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_305 = (
            l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_3_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_307 = torch.nn.functional.relu(sp_306, inplace=True)
        sp_306 = None
        out_83 = torch.cat(
            [sp_283, sp_287, sp_291, sp_295, sp_299, sp_303, sp_307, getitem_87], 1
        )
        sp_283 = sp_287 = sp_291 = sp_295 = sp_299 = sp_303 = sp_307 = getitem_87 = None
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
        split_11 = torch.functional.split(out_90, 56, 1)
        out_90 = None
        sp_308 = split_11[0]
        getitem_89 = split_11[1]
        getitem_90 = split_11[2]
        getitem_91 = split_11[3]
        getitem_92 = split_11[4]
        getitem_93 = split_11[5]
        getitem_94 = split_11[6]
        getitem_95 = split_11[7]
        split_11 = None
        sp_309 = torch.conv2d(
            sp_308,
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_308 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_310 = torch.nn.functional.batch_norm(
            sp_309,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_309 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_311 = torch.nn.functional.relu(sp_310, inplace=True)
        sp_310 = None
        sp_312 = sp_311 + getitem_89
        getitem_89 = None
        sp_313 = torch.conv2d(
            sp_312,
            l_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_312 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_314 = torch.nn.functional.batch_norm(
            sp_313,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_313 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_315 = torch.nn.functional.relu(sp_314, inplace=True)
        sp_314 = None
        sp_316 = sp_315 + getitem_90
        getitem_90 = None
        sp_317 = torch.conv2d(
            sp_316,
            l_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_316 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_318 = torch.nn.functional.batch_norm(
            sp_317,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_317 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_319 = torch.nn.functional.relu(sp_318, inplace=True)
        sp_318 = None
        sp_320 = sp_319 + getitem_91
        getitem_91 = None
        sp_321 = torch.conv2d(
            sp_320,
            l_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_320 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_322 = torch.nn.functional.batch_norm(
            sp_321,
            l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_321 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_323 = torch.nn.functional.relu(sp_322, inplace=True)
        sp_322 = None
        sp_324 = sp_323 + getitem_92
        getitem_92 = None
        sp_325 = torch.conv2d(
            sp_324,
            l_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_324 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_326 = torch.nn.functional.batch_norm(
            sp_325,
            l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_325 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_327 = torch.nn.functional.relu(sp_326, inplace=True)
        sp_326 = None
        sp_328 = sp_327 + getitem_93
        getitem_93 = None
        sp_329 = torch.conv2d(
            sp_328,
            l_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_328 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_330 = torch.nn.functional.batch_norm(
            sp_329,
            l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_329 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_331 = torch.nn.functional.relu(sp_330, inplace=True)
        sp_330 = None
        sp_332 = sp_331 + getitem_94
        getitem_94 = None
        sp_333 = torch.conv2d(
            sp_332,
            l_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_332 = (
            l_self_modules_layer3_modules_4_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_334 = torch.nn.functional.batch_norm(
            sp_333,
            l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_333 = (
            l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_4_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_335 = torch.nn.functional.relu(sp_334, inplace=True)
        sp_334 = None
        out_91 = torch.cat(
            [sp_311, sp_315, sp_319, sp_323, sp_327, sp_331, sp_335, getitem_95], 1
        )
        sp_311 = sp_315 = sp_319 = sp_323 = sp_327 = sp_331 = sp_335 = getitem_95 = None
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
        split_12 = torch.functional.split(out_98, 56, 1)
        out_98 = None
        sp_336 = split_12[0]
        getitem_97 = split_12[1]
        getitem_98 = split_12[2]
        getitem_99 = split_12[3]
        getitem_100 = split_12[4]
        getitem_101 = split_12[5]
        getitem_102 = split_12[6]
        getitem_103 = split_12[7]
        split_12 = None
        sp_337 = torch.conv2d(
            sp_336,
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_336 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_338 = torch.nn.functional.batch_norm(
            sp_337,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_337 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_339 = torch.nn.functional.relu(sp_338, inplace=True)
        sp_338 = None
        sp_340 = sp_339 + getitem_97
        getitem_97 = None
        sp_341 = torch.conv2d(
            sp_340,
            l_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_340 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_342 = torch.nn.functional.batch_norm(
            sp_341,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_341 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_343 = torch.nn.functional.relu(sp_342, inplace=True)
        sp_342 = None
        sp_344 = sp_343 + getitem_98
        getitem_98 = None
        sp_345 = torch.conv2d(
            sp_344,
            l_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_344 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_346 = torch.nn.functional.batch_norm(
            sp_345,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_345 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_347 = torch.nn.functional.relu(sp_346, inplace=True)
        sp_346 = None
        sp_348 = sp_347 + getitem_99
        getitem_99 = None
        sp_349 = torch.conv2d(
            sp_348,
            l_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_348 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_350 = torch.nn.functional.batch_norm(
            sp_349,
            l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_349 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_351 = torch.nn.functional.relu(sp_350, inplace=True)
        sp_350 = None
        sp_352 = sp_351 + getitem_100
        getitem_100 = None
        sp_353 = torch.conv2d(
            sp_352,
            l_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_352 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_354 = torch.nn.functional.batch_norm(
            sp_353,
            l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_353 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_355 = torch.nn.functional.relu(sp_354, inplace=True)
        sp_354 = None
        sp_356 = sp_355 + getitem_101
        getitem_101 = None
        sp_357 = torch.conv2d(
            sp_356,
            l_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_356 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_358 = torch.nn.functional.batch_norm(
            sp_357,
            l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_357 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_359 = torch.nn.functional.relu(sp_358, inplace=True)
        sp_358 = None
        sp_360 = sp_359 + getitem_102
        getitem_102 = None
        sp_361 = torch.conv2d(
            sp_360,
            l_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_360 = (
            l_self_modules_layer3_modules_5_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_362 = torch.nn.functional.batch_norm(
            sp_361,
            l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_361 = (
            l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer3_modules_5_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_363 = torch.nn.functional.relu(sp_362, inplace=True)
        sp_362 = None
        out_99 = torch.cat(
            [sp_339, sp_343, sp_347, sp_351, sp_355, sp_359, sp_363, getitem_103], 1
        )
        sp_339 = (
            sp_343
        ) = sp_347 = sp_351 = sp_355 = sp_359 = sp_363 = getitem_103 = None
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
        split_13 = torch.functional.split(out_106, 112, 1)
        out_106 = None
        sp_364 = split_13[0]
        sp_368 = split_13[1]
        sp_372 = split_13[2]
        sp_376 = split_13[3]
        sp_380 = split_13[4]
        sp_384 = split_13[5]
        sp_388 = split_13[6]
        getitem_111 = split_13[7]
        split_13 = None
        sp_365 = torch.conv2d(
            sp_364,
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_364 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_366 = torch.nn.functional.batch_norm(
            sp_365,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_365 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_367 = torch.nn.functional.relu(sp_366, inplace=True)
        sp_366 = None
        sp_369 = torch.conv2d(
            sp_368,
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_368 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_370 = torch.nn.functional.batch_norm(
            sp_369,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_369 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_371 = torch.nn.functional.relu(sp_370, inplace=True)
        sp_370 = None
        sp_373 = torch.conv2d(
            sp_372,
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_372 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_374 = torch.nn.functional.batch_norm(
            sp_373,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_373 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_375 = torch.nn.functional.relu(sp_374, inplace=True)
        sp_374 = None
        sp_377 = torch.conv2d(
            sp_376,
            l_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_376 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_378 = torch.nn.functional.batch_norm(
            sp_377,
            l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_377 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_379 = torch.nn.functional.relu(sp_378, inplace=True)
        sp_378 = None
        sp_381 = torch.conv2d(
            sp_380,
            l_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_380 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_382 = torch.nn.functional.batch_norm(
            sp_381,
            l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_381 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_383 = torch.nn.functional.relu(sp_382, inplace=True)
        sp_382 = None
        sp_385 = torch.conv2d(
            sp_384,
            l_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_384 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_386 = torch.nn.functional.batch_norm(
            sp_385,
            l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_385 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_387 = torch.nn.functional.relu(sp_386, inplace=True)
        sp_386 = None
        sp_389 = torch.conv2d(
            sp_388,
            l_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_,
            None,
            (2, 2),
            (1, 1),
            (1, 1),
            1,
        )
        sp_388 = (
            l_self_modules_layer4_modules_0_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_390 = torch.nn.functional.batch_norm(
            sp_389,
            l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_389 = (
            l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_0_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_391 = torch.nn.functional.relu(sp_390, inplace=True)
        sp_390 = None
        avg_pool2d_3 = torch._C._nn.avg_pool2d(getitem_111, 3, 2, 1, False, True, None)
        getitem_111 = None
        out_107 = torch.cat(
            [sp_367, sp_371, sp_375, sp_379, sp_383, sp_387, sp_391, avg_pool2d_3], 1
        )
        sp_367 = (
            sp_371
        ) = sp_375 = sp_379 = sp_383 = sp_387 = sp_391 = avg_pool2d_3 = None
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
        split_14 = torch.functional.split(out_114, 112, 1)
        out_114 = None
        sp_392 = split_14[0]
        getitem_113 = split_14[1]
        getitem_114 = split_14[2]
        getitem_115 = split_14[3]
        getitem_116 = split_14[4]
        getitem_117 = split_14[5]
        getitem_118 = split_14[6]
        getitem_119 = split_14[7]
        split_14 = None
        sp_393 = torch.conv2d(
            sp_392,
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_392 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_394 = torch.nn.functional.batch_norm(
            sp_393,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_393 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_395 = torch.nn.functional.relu(sp_394, inplace=True)
        sp_394 = None
        sp_396 = sp_395 + getitem_113
        getitem_113 = None
        sp_397 = torch.conv2d(
            sp_396,
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_396 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_398 = torch.nn.functional.batch_norm(
            sp_397,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_397 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_399 = torch.nn.functional.relu(sp_398, inplace=True)
        sp_398 = None
        sp_400 = sp_399 + getitem_114
        getitem_114 = None
        sp_401 = torch.conv2d(
            sp_400,
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_400 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_402 = torch.nn.functional.batch_norm(
            sp_401,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_401 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_403 = torch.nn.functional.relu(sp_402, inplace=True)
        sp_402 = None
        sp_404 = sp_403 + getitem_115
        getitem_115 = None
        sp_405 = torch.conv2d(
            sp_404,
            l_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_404 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_406 = torch.nn.functional.batch_norm(
            sp_405,
            l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_405 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_407 = torch.nn.functional.relu(sp_406, inplace=True)
        sp_406 = None
        sp_408 = sp_407 + getitem_116
        getitem_116 = None
        sp_409 = torch.conv2d(
            sp_408,
            l_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_408 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_410 = torch.nn.functional.batch_norm(
            sp_409,
            l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_409 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_411 = torch.nn.functional.relu(sp_410, inplace=True)
        sp_410 = None
        sp_412 = sp_411 + getitem_117
        getitem_117 = None
        sp_413 = torch.conv2d(
            sp_412,
            l_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_412 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_414 = torch.nn.functional.batch_norm(
            sp_413,
            l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_413 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_415 = torch.nn.functional.relu(sp_414, inplace=True)
        sp_414 = None
        sp_416 = sp_415 + getitem_118
        getitem_118 = None
        sp_417 = torch.conv2d(
            sp_416,
            l_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_416 = (
            l_self_modules_layer4_modules_1_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_418 = torch.nn.functional.batch_norm(
            sp_417,
            l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_417 = (
            l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_1_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_419 = torch.nn.functional.relu(sp_418, inplace=True)
        sp_418 = None
        out_115 = torch.cat(
            [sp_395, sp_399, sp_403, sp_407, sp_411, sp_415, sp_419, getitem_119], 1
        )
        sp_395 = (
            sp_399
        ) = sp_403 = sp_407 = sp_411 = sp_415 = sp_419 = getitem_119 = None
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
        split_15 = torch.functional.split(out_122, 112, 1)
        out_122 = None
        sp_420 = split_15[0]
        getitem_121 = split_15[1]
        getitem_122 = split_15[2]
        getitem_123 = split_15[3]
        getitem_124 = split_15[4]
        getitem_125 = split_15[5]
        getitem_126 = split_15[6]
        getitem_127 = split_15[7]
        split_15 = None
        sp_421 = torch.conv2d(
            sp_420,
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_420 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_0_parameters_weight_
        ) = None
        sp_422 = torch.nn.functional.batch_norm(
            sp_421,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_421 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_0_parameters_bias_
        ) = None
        sp_423 = torch.nn.functional.relu(sp_422, inplace=True)
        sp_422 = None
        sp_424 = sp_423 + getitem_121
        getitem_121 = None
        sp_425 = torch.conv2d(
            sp_424,
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_424 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_1_parameters_weight_
        ) = None
        sp_426 = torch.nn.functional.batch_norm(
            sp_425,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_425 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_1_parameters_bias_
        ) = None
        sp_427 = torch.nn.functional.relu(sp_426, inplace=True)
        sp_426 = None
        sp_428 = sp_427 + getitem_122
        getitem_122 = None
        sp_429 = torch.conv2d(
            sp_428,
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_428 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_2_parameters_weight_
        ) = None
        sp_430 = torch.nn.functional.batch_norm(
            sp_429,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_429 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_2_parameters_bias_
        ) = None
        sp_431 = torch.nn.functional.relu(sp_430, inplace=True)
        sp_430 = None
        sp_432 = sp_431 + getitem_123
        getitem_123 = None
        sp_433 = torch.conv2d(
            sp_432,
            l_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_432 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_3_parameters_weight_
        ) = None
        sp_434 = torch.nn.functional.batch_norm(
            sp_433,
            l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_433 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_3_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_3_parameters_bias_
        ) = None
        sp_435 = torch.nn.functional.relu(sp_434, inplace=True)
        sp_434 = None
        sp_436 = sp_435 + getitem_124
        getitem_124 = None
        sp_437 = torch.conv2d(
            sp_436,
            l_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_436 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_4_parameters_weight_
        ) = None
        sp_438 = torch.nn.functional.batch_norm(
            sp_437,
            l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_437 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_4_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_4_parameters_bias_
        ) = None
        sp_439 = torch.nn.functional.relu(sp_438, inplace=True)
        sp_438 = None
        sp_440 = sp_439 + getitem_125
        getitem_125 = None
        sp_441 = torch.conv2d(
            sp_440,
            l_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_440 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_5_parameters_weight_
        ) = None
        sp_442 = torch.nn.functional.batch_norm(
            sp_441,
            l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_441 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_5_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_5_parameters_bias_
        ) = None
        sp_443 = torch.nn.functional.relu(sp_442, inplace=True)
        sp_442 = None
        sp_444 = sp_443 + getitem_126
        getitem_126 = None
        sp_445 = torch.conv2d(
            sp_444,
            l_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_,
            None,
            (1, 1),
            (1, 1),
            (1, 1),
            1,
        )
        sp_444 = (
            l_self_modules_layer4_modules_2_modules_convs_modules_6_parameters_weight_
        ) = None
        sp_446 = torch.nn.functional.batch_norm(
            sp_445,
            l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_,
            l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_,
            l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_,
            l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_,
            False,
            0.1,
            1e-05,
        )
        sp_445 = (
            l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_mean_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_6_buffers_running_var_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_weight_
        ) = (
            l_self_modules_layer4_modules_2_modules_bns_modules_6_parameters_bias_
        ) = None
        sp_447 = torch.nn.functional.relu(sp_446, inplace=True)
        sp_446 = None
        out_123 = torch.cat(
            [sp_423, sp_427, sp_431, sp_435, sp_439, sp_443, sp_447, getitem_127], 1
        )
        sp_423 = (
            sp_427
        ) = sp_431 = sp_435 = sp_439 = sp_443 = sp_447 = getitem_127 = None
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
